#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import dheera_ai


def test_update_model_cost():
    try:
        dheera_ai.register_model(
            {
                "gpt-4": {
                    "max_tokens": 8192,
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00006,
                    "dheera_ai_provider": "openai",
                    "mode": "chat",
                },
            }
        )
        assert dheera_ai.model_cost["gpt-4"]["input_cost_per_token"] == 0.00002
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_update_model_cost()


def test_update_model_cost_map_url():
    try:
        dheera_ai.register_model(
            model_cost="https://raw.githubusercontent.com/BerriAI/dheera_ai/main/model_prices_and_context_window.json"
        )
        assert dheera_ai.model_cost["gpt-4"]["input_cost_per_token"] == 0.00003
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_update_model_cost_map_url()


def test_update_model_cost_via_completion():
    try:
        response = dheera_ai.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            input_cost_per_token=0.3,
            output_cost_per_token=0.4,
        )
        print(
            f"dheera_ai.model_cost for gpt-3.5-turbo: {dheera_ai.model_cost['gpt-3.5-turbo']}"
        )
        assert dheera_ai.model_cost["gpt-3.5-turbo"]["input_cost_per_token"] == 0.3
        assert dheera_ai.model_cost["gpt-3.5-turbo"]["output_cost_per_token"] == 0.4
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


test_update_model_cost_via_completion()
