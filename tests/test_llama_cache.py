# pylint: disable=redefined-outer-name,missing-function-docstring,missing-module-docstring
import ctypes
import os
from random import seed
import tempfile

import numpy as np
import numpy.typing as npt
import pytest

from llama_cpp.llama import Llama, LlamaState
from llama_cpp.llama_cache import LlamaStaticDiskCache, StateReloadError


def _get_logits(model: Llama) -> npt.NDArray:
    """
    Helper method to get logits and put into correct shape.

    (Model returns the non-zero logits in batch, which may be different
    depending on whether or not `logits_all` is True).

    Makes a copy so that not limited by the lifetime of the CtypesArray.
    """
    # pylint: disable=protected-access
    logits: ctypes.Array[ctypes.c_float] = model._ctx.get_logits_ith(-1)

    # Return None if falsy (NULL)
    if not logits:
        return None

    num_rows = 1
    num_cols = model.n_vocab()

    logits_np: npt.NDArray = np.ctypeslib.as_array(
        logits, shape=(num_rows, num_cols)
    ).copy()

    return logits_np


# Have to be careful to reset to good state when testing, but don't want to
# recreate model each time.
def model_factory(**kwargs) -> Llama:
    model_filename = os.getenv("LLAMA_TEST_MODEL")
    if not model_filename:
        pytest.skip("LLAMA_TEST_MODEL environment variable is not set")
        return

    model_filename = os.path.expanduser(model_filename)

    default_args = dict(
        n_ctx=2_048,
        n_gpu_layers=0,
        offload_kqv=False,
        n_batch=512,
        embedding=False,
        # Warning - since now uses llama.cpp sampler, no longer generates
        # logits for each generated token unless this is True.
        logits_all=False,
        verbose=False,
    )

    default_args.update(kwargs)

    test_model = Llama(model_filename, **default_args)

    return test_model


@pytest.fixture(scope="module")
def system_prompt() -> str:
    return r"""
You are an advanced intelligence "Hal" aboard a spaceship. You are required to
act as the primary interface between the ship and its crew. You can:
* Provide information on the current status of the ship
* Turn on/off the lights in the crew quarters
* Open/close the airlocks

Respond in a terse, professional manner. Do not engage in casual conversation.

The current state of the ship is:
* Airlocks: closed
* Lights: on
* Oxygen levels: normal
""".strip()


@pytest.fixture(scope="module")
def user_prompt() -> str:
    return "Hal, please open the airlocks."


@pytest.fixture(scope="module")
def small_model(system_prompt: str, user_prompt: str):
    """
    Create model and and run prompt through it to make sure has logits for last
    token generation (internally).

    Logits on numpy array will be all zeros since `logits_all` is False.
    """
    model = model_factory()

    # Ingest prompt and create completion so that will have some state.
    # Last token of prompt + all tokens of generated completion will have
    # non-zero logits.
    _ = model.create_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=1234,
    )

    assert model.n_tokens > 0

    # Have logits for last token
    logits_np = _get_logits(model)
    assert logits_np.shape == (1, model.n_vocab())
    assert ~(logits_np == 0.0).all()

    assert (model.scores == 0.0).all()

    return model


@pytest.fixture(scope="module")
def small_model_with_logits(system_prompt: str, user_prompt: str) -> Llama:
    """
    Create model with logits_all=True, needed for testing building/reloading cache
    when Python-land logits are needed.
    """
    model = model_factory(logits_all=True)

    # Ingest prompt and create completion so that will have some state.
    # Last token of prompt + all tokens of generated completion will have
    # non-zero logits.
    _ = model.create_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=1234,
    )

    assert model.n_tokens > 0

    # Have logits for last token
    logits_np = _get_logits(model)
    assert logits_np.shape == (1, model.n_vocab())
    assert ~(logits_np == 0.0).all()

    assert ~(model.scores == 0.0).all()

    return model


@pytest.fixture(scope="module")
def llama_state(small_model) -> LlamaState:
    state = small_model.save_state()
    # Clear scores so that can test reloading from cache without them.
    state.scores = None
    return state


def test_reload_from_cache_state_success(small_model, llama_state: LlamaState):
    current_state = small_model.save_state()
    old_logits = _get_logits(small_model)

    # Create blank model
    new_model = model_factory()
    LlamaStaticDiskCache.reload_from_cache_state(new_model, llama_state)

    assert (current_state.input_ids == new_model.input_ids).all()

    assert current_state.n_tokens == new_model.n_tokens

    # pylint: disable=protected-access
    assert current_state.seed == new_model._seed

    new_logits = _get_logits(new_model)

    # Logits for last token should match, others may not if n_batch < n_tokens
    assert (new_logits == old_logits).all()


def test_reload_from_cache_state_state_reload_error(small_model, llama_state):
    small_model.context_params.logits_all = True
    small_model.context_params.embeddings = True
    try:
        with pytest.raises(StateReloadError):
            LlamaStaticDiskCache.reload_from_cache_state(small_model, llama_state)
    finally:
        small_model.context_params.logits_all = False
        small_model.context_params.embeddings = False


def test_disk_cache_e2e(small_model: Llama):
    prompts = ["this is a test prompt", "and this is another test prompt"]
    capacity_bytes = 2 << 30

    small_model.reset()
    # This is a weird thing to reset, but input_ids > n_tokens are not
    # significant (like a scratchpad), left over if had previous prompt that
    # was longer.
    #
    # Reset for ease of comparison later.
    small_model.input_ids[:] = 0

    with tempfile.TemporaryDirectory() as cache_dir:
        cache = LlamaStaticDiskCache.build_cache(
            cache_dir=cache_dir,
            prompts=prompts,
            model=small_model,
            capacity_bytes=capacity_bytes,
            add_bos=True,
            seed=1234,
            save_logits=False,
        )

        for p in prompts:
            key = tuple(
                small_model.tokenize(p.encode("utf-8"), add_bos=True, special=True)
            )
            assert key in cache
            state = cache[key]
            assert ~(state.input_ids == 0).all()
            assert state is not None
            assert (
                state.scores is None
            ), "Logits should not be stored when save_logits=False and model doesn't require them."

            small_model.reset()
            small_model.input_ids[:] = 0
            small_model.eval(key)

            state2 = small_model.save_state()
            assert state2.n_tokens == state.n_tokens
            assert ~(state2.input_ids == 0).all()
            assert (state2.input_ids == state.input_ids).all()

            last_logits = _get_logits(small_model)

            LlamaStaticDiskCache.reload_from_cache_state(small_model, state)

            last_logits2 = _get_logits(small_model)

            assert (last_logits == last_logits2).all()


def test_cache_save_reload_scores_when_needed(
    small_model_with_logits: Llama,
):
    """
    When model requires it, can reload from state with scores.
    """
    model_state_before_reload = small_model_with_logits.save_state()

    test_prompt = "this is a test prompt"
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = LlamaStaticDiskCache.build_cache(
            cache_dir=cache_dir,
            prompts=[test_prompt],
            model=small_model_with_logits,
            capacity_bytes=2 << 30,
            add_bos=True,
            seed=1234,
            save_logits=True,
        )

        llama_state = small_model_with_logits.save_state()
        cur_scores = llama_state.scores.copy()
        assert ~(cur_scores == 0.0).all()
        assert llama_state.n_tokens > 0

        try:
            state_from_cache = cache[
                tuple(llama_state.input_ids[: llama_state.n_tokens].tolist())
            ]
            assert state_from_cache.scores is not None, "Scores should be saved."
            LlamaStaticDiskCache.reload_from_cache_state(
                small_model_with_logits, state_from_cache
            )

            assert state_from_cache.n_tokens == small_model_with_logits.n_tokens
            # pylint: disable=protected-access
            assert state_from_cache.seed == small_model_with_logits._seed

            # Do I have to limit these to n_tokens?
            assert (state_from_cache.input_ids == llama_state.input_ids).all()
            assert (
                cur_scores
                == small_model_with_logits.scores[: small_model_with_logits.n_tokens]
            ).all(), "Reloaded scores should match"
        finally:
            # Reset if re-used in later tests
            small_model_with_logits.load_state(model_state_before_reload)


def test_cache_reload_errors_when_requires_scores_and_state_doesnt_have_it(
    small_model: Llama, llama_state: LlamaState
):
    """
    If model requires logits for sampling and state doesn't have it, should raise error.
    """
    old_state_scores = (
        llama_state.scores.copy()
        if llama_state.scores is not None
        else llama_state.scores
    )
    try:
        small_model.context_params.logits_all = True
        llama_state.scores = None

        with pytest.raises(StateReloadError):
            LlamaStaticDiskCache.reload_from_cache_state(small_model, llama_state)
    finally:
        small_model.context_params.logits_all = False
        llama_state.scores = old_state_scores


# pylint: disable=invalid-name
def test_cache_errors_when_save_logits_False_but_model_requires(small_model: Llama):
    """
    If model requires logits but save_logits is False, should raise error.
    """

    try:
        small_model.context_params.logits_all = True
        with pytest.raises(ValueError):
            with tempfile.TemporaryDirectory() as cache_dir:
                LlamaStaticDiskCache.build_cache(
                    cache_dir=cache_dir,
                    prompts=["this is a test prompt"],
                    model=small_model,
                    capacity_bytes=2 << 30,
                    add_bos=True,
                    seed=1234,
                    save_logits=False,
                )
    finally:
        small_model.context_params.logits_all = False
