import time
from typing import Any, Optional

import dheera_ai
from dheera_ai import CustomLLM, ImageObject, ImageResponse, completion, get_llm_provider
from dheera_ai.llms.custom_httpx.http_handler import AsyncHTTPHandler
from dheera_ai.types.utils import ModelResponse


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> ModelResponse:
        return dheera_ai.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> dheera_ai.ModelResponse:
        return dheera_ai.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


my_custom_llm = MyCustomLLM()
