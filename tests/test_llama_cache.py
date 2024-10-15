import os
import tempfile

import pytest

from llama_cpp.llama import Llama, LlamaState
from llama_cpp.llama_cache import LlamaStaticDiskCache, StateReloadError


@pytest.fixture
def small_model():
    model_filename = os.getenv("LLAMA_TEST_MODEL")
    if not model_filename:
        pytest.skip("LLAMA_TEST_MODEL environment variable is not set")
        return

    model_filename = os.path.expanduser(model_filename)

    test_model = Llama(
        model_filename,
        n_ctx=2_048,
        n_gpu_layers=0,
        offload_kqv=False,
        n_batch=512,
        embedding=False,
        verbose=False,
    )

    system_prompt = r"""
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

    user_prompt = "Hal, please open the airlocks."

    # Ingest prompt and create completion so that will have some state.
    # Last token of prompt + all tokens of generated completion will have
    # non-zero logits.
    _ = test_model.create_chat_completion(
        [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt},
        ],
        seed=1234,
    )

    assert test_model.n_tokens > 0

    # Have at least some scores, and last entry is non-zero
    assert ~(test_model.scores == 0).all()
    # pylint: disable=protected-access
    assert (test_model._scores[-1, :] != 0.0).all()

    return test_model


@pytest.fixture
def llama_state(small_model) -> LlamaState:
    state = small_model.save_state()
    return state


def test_reload_from_cache_state_success(small_model, llama_state: LlamaState):
    current_state = small_model.save_state()
    old_score = small_model.scores.copy()

    LlamaStaticDiskCache.reload_from_cache_state(small_model, llama_state)
    new_state = small_model.save_state()
    new_score = small_model.scores.copy()

    assert (current_state.input_ids == new_state.input_ids).all()

    assert current_state.n_tokens == new_state.n_tokens

    # Logits for last token should match, others may not if n_batch < n_tokens
    assert (
        old_score[small_model.n_tokens - 1, :] == new_score[small_model.n_tokens - 1, :]
    ).all()


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

            small_model.reset()
            small_model.input_ids[:] = 0
            small_model.eval(key)

            state2 = small_model.save_state()
            assert state2.n_tokens == state.n_tokens
            assert ~(state2.input_ids == 0).all()
            assert (state2.input_ids == state.input_ids).all()

            last_logits = small_model.scores[small_model.n_tokens - 1, :]

            LlamaStaticDiskCache.reload_from_cache_state(small_model, state)

            last_logits2 = small_model.scores[small_model.n_tokens - 1, :]

            assert (last_logits == last_logits2).all()
