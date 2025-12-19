from typing import Any, Dict, List, Literal, Optional, Union

import dheera_ai
from dheera_ai._logging import verbose_proxy_logger
from dheera_ai.caching.caching import DualCache
from dheera_ai.integrations.custom_guardrail import CustomGuardrail
from dheera_ai.proxy._types import UserAPIKeyAuth
from dheera_ai.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from dheera_ai.types.utils import CallTypesLiteral


class myCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        # In this guardrail, if a user inputs `dheera_ai` we will mask it and then send it to the LLM
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "dheera_ai" in _content.lower():
                        _content = _content.replace("dheera_ai", "********")
                        message["content"] = _content

        verbose_proxy_logger.debug(
            "async_pre_call_hook: Message after masking %s", _messages
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """

        # this works the same as async_pre_call_hook, but just runs in parallel as the LLM API Call
        # In this guardrail, if a user inputs `dheera_ai` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "dheera_ai" in _content.lower():
                        raise ValueError("Guardrail failed words - `dheera_ai` detected")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response

        If a response contains the word "coffee" -> we will raise an exception
        """
        verbose_proxy_logger.debug("async_pre_call_hook response: %s", response)
        if isinstance(response, dheera_ai.ModelResponse):
            for choice in response.choices:
                if isinstance(choice, dheera_ai.Choices):
                    verbose_proxy_logger.debug("async_pre_call_hook choice: %s", choice)
                    if (
                        choice.message.content
                        and isinstance(choice.message.content, str)
                        and "coffee" in choice.message.content
                    ):
                        raise ValueError("Guardrail failed Coffee Detected")
