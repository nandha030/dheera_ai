#### What this tests ####
#    This tests the model alias mapping - if user passes in an alias, and has set an alias, set it to the actual value

import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import dheera_ai
from dheera_ai import completion, embedding

dheera_ai.set_verbose = True

model_alias_map = {"good-model": "groq/llama-3.1-8b-instant"}


def test_model_alias_map(caplog):
    try:
        dheera_ai.model_alias_map = model_alias_map
        response = completion(
            "good-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            top_p=0.1,
            temperature=0.01,
            max_tokens=10,
        )
        print(response.model)

        captured_logs = [rec.levelname for rec in caplog.records]

        for log in captured_logs:
            assert "ERROR" not in log

        assert "llama-3.1-8b-instant" in response.model
    except dheera_ai.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_model_alias_map()
