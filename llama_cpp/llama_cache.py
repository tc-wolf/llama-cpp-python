import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import diskcache
import pytrie

import llama_cpp.llama

from .llama_types import *


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

    Still using diskcache.Cache for underlying cache, but tries to avoid using
    pickle.dumps when writing LlamaState (which adds overhead).

    Instead of storing serialized LlamaState directly, just store bytes.
    """

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        self.cache = diskcache.Cache(cache_dir, size_limit=capacity_bytes)
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
    ) -> "LlamaStaticDiskCache":
        """
        Using model passed in, evaluates each prompt and stores LlamaState in cache.

        Returns a new LlamaStaticDiskCache instance with cache at cache_dir.
        """
        cache = LlamaStaticDiskCache(cache_dir, capacity_bytes)

        for p in prompts:
            model.reset()
            # Special tokens == control characters like in ChatML
            toks = model.tokenize(p.encode("utf-8"), add_bos=True, special=True)
            print("LlamaStaticDiskCache.build_cache: eval", file=sys.stderr)
            model.eval(toks)
            state = model.save_state()
            cache._private_setitem(toks, state)

        # Set up Trie for efficient prefix search
        for key in cache.cache.iterkeys():
            cache.keys[key] = None

        return cache

    def _find_longest_prefix_key(self, key: Tuple[int]) -> Tuple[int] | None:
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
        value = self.cache[longest_prefix]
        return value

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        # Should this just be a warning?
        raise ValueError("Cannot set items in a static cache")
