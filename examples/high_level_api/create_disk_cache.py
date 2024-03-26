"""
Creates static disk cache give a dataset of snippets to use as contexts for a RAG application.

Background: Want to embed the fixed system prompt + first context and then store on disk. 
That way, when a question is asked and the first context is provided, can look up the
KV cache to find the prompt that matches the prompt tokens (including first context). 

This should save on prompt ingestion time, decreasing time to first token.
"""

import argparse
import os
import pathlib

import pandas as pd
import tqdm

from llama_cpp.llama import Llama
from llama_cpp.llama_cache import LlamaStaticDiskCache
from llama_cpp.llama_chat_format import format_nekomata

# Add additional formatters here as desired so that can swap out models.
CHAT_FORMATTER_MAP = {
    "rinna/nekomata-7b-instruction-gguf": format_nekomata,
}


def combine_question_ctx_nekomata(question, contexts):
    """
    Formats question and contexts for nekomata-7b.
    """
    output = ""
    for i, context in enumerate(contexts):
        output += f"- 資料{i+1}: '{context}'\n"

    output += "\n"

    output += question

    return output


# How to combine contexts + user question when creating a *full* prompt
CONTEXT_QUESTION_FORMATTER_MAP = {
    "rinna/nekomata-7b-instruction-gguf": combine_question_ctx_nekomata,
}

DEFAULT_SYSTEM_PROMPT = """
You are a virtual assistant.  You will be provided with contexts and a user
question.  Your job is to answer a user's question faithfully and concisely.

If the context provided does not contain enough information to answer the question,
respond with "I don't know" - do not make up information.  If you are helpful and provide
accurate information, you will be provided with a $10,000 bonus. If you provide inaccurate
information, unhelpful responses, or information not grounded in the context
provided, you will be penalized $10,000 and fired - into the Sun.
""".strip()


def _create_nekomata_prompt_prefix(
    context: str, system_prompt=DEFAULT_SYSTEM_PROMPT
) -> str:
    """
    Override this if using a different model.

    This provides a partially formatted prompt for the Nekomata model.
    It passes in the system prompt and the first context to the model,
    but not the question or prompt for assistant.
    """

    return """
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
{system_prompt}

### 入力:
{input}""".format(
        system_prompt=system_prompt, input=f"- 資料1: '{context.strip()}'\n"
    ).lstrip(
        "\n"
    )


PARTIAL_PROMPT_MODEL_MAP = {
    "rinna/nekomata-7b-instruction-gguf": _create_nekomata_prompt_prefix,
}


def main(args: argparse.Namespace):
    dataset_path: pathlib.Path = args.dataset
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"

    dataset = pd.read_csv(str(dataset_path))

    snippets = dataset.loc[:, args.column_name].tolist()

    model = Llama.from_pretrained(
        args.model,
        filename=args.model_filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=-1,
        n_batch=1,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads,
        verbose=False,
    )

    prompt_formatter_func = PARTIAL_PROMPT_MODEL_MAP[args.model]

    # Have to format such that includes system prompt and the context
    snippets = [prompt_formatter_func(context) for context in snippets]

    cache = LlamaStaticDiskCache.build_cache(args.output, tqdm.tqdm(snippets), model)
    snippet_tokens = model.tokenize(
        snippets[0].encode("utf-8"), add_bos=True, special=True
    )
    assert snippet_tokens in cache, "First snippet not in cache"

    # pylint: disable=protected-access
    cache_prefix_tokens = cache._find_longest_prefix_key(snippet_tokens)

    assert cache_prefix_tokens == tuple(
        snippet_tokens
    ), "Expected all snippet tokens to be in cache"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=pathlib.Path,
        required=True,
        help="Path to serialized dataframe with snippets to use",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="rinna/nekomata-7b-instruction-gguf",
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        default="*Q4_K_M.gguf",
        help="Name of model weights file to load from repo - may contain wildcards (like '*Q4_K_M.gguf')",
    )

    parser.add_argument(
        "--n-ctx",
        type=int,
        required=True,
        help="Context size (in tokens) - must be fixed for KV cache",
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        default=os.cpu_count(),
        help="Number of threads to use for inference + batch processing",
    )

    parser.add_argument(
        "--column-name",
        type=str,
        default="snippets",
        help="Column name identifying snippets to use as contexts",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="static_cache",
        help="Output directory for static cache",
    )

    args = parser.parse_args()

    chat_formatter = CHAT_FORMATTER_MAP[args.model]
    question_ctx_combiner = CONTEXT_QUESTION_FORMATTER_MAP[args.model]

    DUMMY_CONTEXTS = [
        "The air speed of an unladen swallow is 24 miles per hour.",
        "Red pandas are not actually pandas, but are more closely related to raccoons.",
        "Directly observing a quantum system can change its state.",
        "The mitochondria is the powerhouse of the cell.",
        "The least common multiple of 6 and 8 is 24.",
    ]

    # Just a quick-and-dirty test so that can verify that a full prompt will contain
    # the partial prompt (and so prefix matching should work)
    def _generate_full_prompt(user_question: str):
        user_msg = question_ctx_combiner(user_question, DUMMY_CONTEXTS)
        msgs = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ]

        full_prompt = chat_formatter(msgs).prompt

        return full_prompt

    question = "What is the velocity of an unladen swallow?"

    complete_prompt = _generate_full_prompt(question)
    partial_context = _create_nekomata_prompt_prefix(DUMMY_CONTEXTS[0])

    if not partial_context in complete_prompt:
        print("Partial context:\n")
        print(partial_context + "\n")
        print("not found in complete prompt:\n")
        print(complete_prompt)
        assert False, "Sanity check failed"

    main(args)
