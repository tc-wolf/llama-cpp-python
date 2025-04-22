import pytest

from llama_cpp.server.settings import ModelSettings
from pydantic import ValidationError

# Required to pass in model name
DUMMY_MODEL_NAME = "foo"


def test_formatted_prompt_path_default_none():
    m = ModelSettings(model=DUMMY_MODEL_NAME)
    assert m.formatted_prompt_path is None


def test_validation_error_if_prompt_path_not_endswith_ndjson():
    with pytest.raises(
        ValidationError, match=r"String should match pattern '.*\\.ndjson\$'"
    ):
        ModelSettings(model=DUMMY_MODEL_NAME, formatted_prompt_path="invalid_path.txt")


def test_formatted_prompt_path_works_if_endswith_ndjson():
    m = ModelSettings(model=DUMMY_MODEL_NAME, formatted_prompt_path="valid_path.ndjson")
    assert m.formatted_prompt_path == "valid_path.ndjson"
