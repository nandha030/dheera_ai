import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import dheera_ai
from dheera_ai.proxy.guardrails.init_guardrails import init_guardrails_v2


def test_guardrails_ai():
    dheera_ai.set_verbose = True
    dheera_ai.guardrail_name_config_map = {}

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "gibberish-guard",
                "dheera_ai_params": {
                    "guardrail": "guardrails_ai",
                    "guard_name": "gibberish_guard",
                    "mode": "post_call",
                },
            }
        ],
        config_file_path="",
    )
