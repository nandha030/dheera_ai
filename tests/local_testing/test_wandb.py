import sys
import os
import io, asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from dheera_ai import completion
import dheera_ai

dheera_ai.num_retries = 3
dheera_ai.success_callback = ["wandb"]
import time
import pytest


def test_wandb_logging_async():
    try:
        dheera_ai.set_verbose = False

        async def _test_langfuse():
            from dheera_ai import Router

            model_list = [
                {  # list of model deployments
                    "model_name": "gpt-3.5-turbo",
                    "dheera_ai_params": {  # params for dheera_ai completion/embedding call
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                }
            ]

            router = Router(model_list=model_list)

            # openai.ChatCompletion.create replacement
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "this is a test with dheera_ai router ?"}
                ],
            )
            print(response)

        response = asyncio.run(_test_langfuse())
        print(f"response: {response}")
    except dheera_ai.Timeout as e:
        pass
    except Exception as e:
        pass


test_wandb_logging_async()


def test_wandb_logging():
    try:
        response = completion(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
    except dheera_ai.Timeout as e:
        pass
    except Exception as e:
        print(e)


# test_wandb_logging()
