import ctypes
import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import diskcache
import numpy as np
import pytrie

import llama_cpp.llama

from .llama_types import *


class StateReloadError(Exception):
    """
    Error for when state from cache cannot be read by current model.
    """


class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_ro(self) -> bool:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    @abstractmethod
    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Sequence[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(
        self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"
    ) -> None:
        raise NotImplementedError

    @classmethod
    def reload_from_cache_state(
        cls, model: "llama_cpp.llama.Llama", state: "llama_cpp.llama.LlamaState"
    ) -> None:
        """
        Reload the state onto the model.  Normally this is done with load_state
        (as state is created with the corresponding `save_state`), but for some
        caches may need special handling as an optimization.

        Throws a StateReloadError if the state is not compatible with the model
        (for example, logits )
        """
        model.load_state(state)


class LlamaRAMCache(BaseLlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[Tuple[int, ...], "llama_cpp.llama.LlamaState"] = (
            OrderedDict()
        )

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    @property
    def is_ro(self) -> bool:
        return False

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key = None
        keys = (
            (k, llama_cpp.llama.Llama.longest_token_prefix(k, key))
            for k in self.cache_state.keys()
        )
        for k, prefix_len in keys:
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]
        self.cache_state[key] = value
        while self.cache_size > self.capacity_bytes and len(self.cache_state) > 0:
            self.cache_state.popitem(last=False)


# Alias for backwards compatibility
LlamaCache = LlamaRAMCache


class LlamaDiskCache(BaseLlamaCache):
    """Cache for a llama.cpp model using disk."""

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        super().__init__(capacity_bytes)
        self.cache = diskcache.Cache(cache_dir)

    @property
    def cache_size(self):
        return int(self.cache.volume())  # type: ignore

    @property
    def is_ro(self) -> bool:
        return False

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        for k in self.cache.iterkeys():  # type: ignore
            prefix_len = llama_cpp.llama.Llama.longest_token_prefix(k, key)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k  # type: ignore
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value: "llama_cpp.llama.LlamaState" = self.cache.pop(_key)  # type: ignore
        # NOTE: This puts an integer as key in cache, which breaks,
        # Llama.longest_token_prefix(k, key) above since k is not a tuple of ints/tokens
        # self.cache.push(_key, side="front")  # type: ignore
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
        key = tuple(key)
        if key in self.cache:
            print("LlamaDiskCache.__setitem__: delete", file=sys.stderr)
            del self.cache[key]
        self.cache[key] = value
        print("LlamaDiskCache.__setitem__: set", file=sys.stderr)
        while self.cache_size > self.capacity_bytes and len(self.cache) > 0:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        print("LlamaDiskCache.__setitem__: trim", file=sys.stderr)


class LlamaStaticDiskCache(BaseLlamaCache):
    """
    Cache that only reads from the cache, doesn't store / overwrite items, and
    doesn't pop from cache.

    Still using diskcache.Cache for underlying cache, but uses a trie to store
    keys so that can more efficiently look for prefixes.

    Want to store C++ state as bytes (from `llama_copy_state_data`), but for now
    still storing LlamaState, because need scores/input_ids/n_tokens so that Python
    code can continue inference.
    """

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        self.cache = diskcache.Cache(
            cache_dir, size_limit=capacity_bytes, cull_limit=0, eviction_policy="none"
        )
        self.capacity_bytes = capacity_bytes
        # Don't want to have to iterate over all keys when doing longest matching prefix search
        self.keys = pytrie.Trie.fromkeys(self.cache.iterkeys())

    @property
    def cache_size(self):
        return int(self.cache.volume())  # type: ignore

    @property
    def is_ro(self) -> bool:
        return True

    def _private_setitem(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        if self.cache_size > self.capacity_bytes:
            # I think it's okay to raise an error here, because only done when building cache anyway.
            raise ValueError("Cache is full, refusing to set more")

        key = tuple(key)
        if key in self.cache:
            print(
                "LlamaStaticDiskCache._private_setitem: delete (overwriting)",
                file=sys.stderr,
            )
            del self.cache[key]

        # This is what diskcache does anyway, eventually want this to be more compact
        print("LlamaStaticDiskCache._private_setitem: set", file=sys.stderr)
        self.cache[key] = pickle.dumps(value, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def build_cache(
        cache_dir: str,
        prompts: Sequence[str],
        model: "llama_cpp.Llama",
        # Same default as LlamaDiskCache, 1 GB
        capacity_bytes: int = 2 << 30,
        seed: Optional[int] = None,
        add_bos=True,
        save_logits: bool = False,
    ) -> "LlamaStaticDiskCache":
        """
        Using model passed in, evaluates each prompt and stores LlamaState in cache.

        Returns a new LlamaStaticDiskCache instance with cache at cache_dir.
        """
        cache = LlamaStaticDiskCache(cache_dir, capacity_bytes)

        for p in prompts:
            if seed:
                model.set_seed(seed)
            # Special tokens == control characters like in ChatML
            toks = model.tokenize(p.encode("utf-8"), add_bos=add_bos, special=True)
            # Will always eval at least one token, same logic as in
            # `Llama.generate` for prefix-match hit.
            # pylint: disable=protected-access
            shared_prefix_len = model.longest_token_prefix(toks[:-1], model._input_ids)
            # Reset to shared prefix length so that don't have to re-eval system prompt
            model.n_tokens = shared_prefix_len
            eval_toks = toks[shared_prefix_len:]
            print("LlamaStaticDiskCache.build_cache: eval", file=sys.stderr)
            model.eval(eval_toks)
            state = model.save_state()

            if not save_logits:
                if (
                    model.context_params.logits_all
                    or model.draft_model is not None
                    or model.context_params.embeddings
                ):
                    # Erroring instead of falling back to just saving with scores
                    raise ValueError(
                        "Cannot save state without logits - model requires logits to sample."
                    )
                state.scores = None

            cache._private_setitem(toks, state)  # pylint: disable=protected-access

        # Set up Trie for efficient prefix search
        for key in cache.cache.iterkeys():
            cache.keys[key] = None

        return cache

    def _find_longest_prefix_key(self, key: Tuple[int]) -> Optional[Tuple[int, ...]]:
        try:
            longest_prefix = self.keys.longest_prefix(key)
            return longest_prefix
        except KeyError:
            return None

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        """
        Only handling exact matches (not prefixes).  Use case is that have some
        prompt + context that want to match against.
        """
        key = tuple(key)
        # Don't worry about KeyError, that's handled by caller
        longest_prefix = self._find_longest_prefix_key(key)
        value: "llama_cpp.llama.LlamaState" = pickle.loads(self.cache[longest_prefix])
        return value

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        # Should this just be a warning?
        raise ValueError("Cannot set items in a static cache")

    @classmethod
    def reload_from_cache_state(
        cls, model: "llama_cpp.llama.Llama", state: "llama_cpp.llama.LlamaState"
    ) -> None:
        """
        Skip reloading logits and set last logits from llama.cpp context struct
        as the scores for last token of prompt.
        """
        # pylint: disable=protected-access

        # Check if model needs logits (draft model, log probs required, etc.)
        need_to_reload_without_scores = (
            # May be overly pessimistic if don't want embeddings for prompt tokens.
            model.context_params.embeddings
            or model.context_params.logits_all
            # Same: is this really a hard requirement? We need token IDs from
            # draft model and all the logits from base model to do verification
            # of candidate tokens, but not for prompt tokens.
            or model.draft_model is not None
        )

        if need_to_reload_without_scores:
            if state.scores is None:
                raise StateReloadError(
                    "Model requires logits to be reloaded, but static cache does not store logits"
                )
            else:
                model.load_state(state)
                return

        # Case where don't need logits from numpy and can just get last-token
        # logits from llama.cpp struct
        model.n_tokens = state.n_tokens
        model.input_ids = state.input_ids.copy()
        model.scores[:] = 0.0

        state_size = state.llama_state_size

        try:
            llama_state_array_type = ctypes.c_uint8 * state_size
            # Have to do from_buffer_copy since LlamaState.llama_state is
            # non-mutable bytes, not mutable bytearray.
            llama_state = llama_state_array_type.from_buffer_copy(state.llama_state)
            reloaded_state_size = llama_cpp.llama_set_state_data(
                model._ctx.ctx, llama_state
            )

            if reloaded_state_size != state_size:
                raise StateReloadError(
                    "Failed to set llama state data - reloaded state size "
                    f"{reloaded_state_size} does not match original size {state_size}"
                )

            # cffi dtype, compatible w/ numpy through ducktyping :scared:
            dtype = llama_cpp.llama_cpp.llama_get_logits_ith.restype._type_

            # If model scores dtype doesn't match dtype from sig, then can't
            # copy it.
            if model.scores.dtype != dtype:
                raise StateReloadError(
                    f"Expected scores to be {dtype} but got "
                    f"{model.scores.dtype} - are you running this in the future? Or the past?"
                )

            # Will have a ValueError for null pointers
            last_position_logits = np.array(
                ctypes.cast(
                    model._ctx.get_logits_ith(-1),
                    ctypes.POINTER(dtype * model.n_vocab()),
                ).contents,
                # Otherwise will be a view into C array on llama.cpp context
                copy=True,
                dtype=dtype,
            )

            model._scores[-1, :] = last_position_logits

        except ValueError as e:
            raise StateReloadError from e
