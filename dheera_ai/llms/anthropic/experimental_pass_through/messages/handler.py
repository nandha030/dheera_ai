"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here
- Ensure requests are logged in the DB - stream + non-stream

"""

import asyncio
import contextvars
from functools import partial
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union

import dheera_ai
from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
from dheera_ai.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from dheera_ai.llms.custom_httpx.http_handler import AsyncHTTPHandler
from dheera_ai.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from dheera_ai.types.llms.anthropic_messages.anthropic_request import AnthropicMetadata
from dheera_ai.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.utils import ProviderConfigManager, client

from ..adapters.handler import DheeraAIMessagesToCompletionTransformationHandler
from .utils import AnthropicMessagesRequestUtils, mock_response

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@client
async def anthropic_messages(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    """
    Async: Make llm api request in Anthropic /messages API spec
    """
    local_vars = locals()
    loop = asyncio.get_event_loop()
    kwargs["is_async"] = True

    func = partial(
        anthropic_messages_handler,
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=thinking,
        tool_choice=tool_choice,
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        api_key=api_key,
        api_base=api_base,
        client=client,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)
    init_response = await loop.run_in_executor(None, func_with_context)

    if asyncio.iscoroutine(init_response):
        response = await init_response
    else:
        response = init_response
    return response


def validate_anthropic_api_metadata(metadata: Optional[Dict] = None) -> Optional[Dict]:
    """
    Validate Anthropic API metadata - This is done to ensure only allowed `metadata` fields are passed to Anthropic API

    If there are any dheera_ai specific metadata fields, use `dheera_ai_metadata` key to pass them.
    """
    if metadata is None:
        return None
    anthropic_metadata_obj = AnthropicMetadata(**metadata)
    return anthropic_metadata_obj.model_dump(exclude_none=True)


def anthropic_messages_handler(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    AnthropicMessagesResponse,
    AsyncIterator[Any],
    Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec
    """
    from dheera_ai.types.utils import LlmProviders

    metadata = validate_anthropic_api_metadata(metadata)

    local_vars = locals()
    is_async = kwargs.pop("is_async", False)
    # Use provided client or create a new one
    dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore

    dheera_ai_params = GenericDheeraAIParams(
        **kwargs,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
    )
    (
        model,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = dheera_ai.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
    )

    if dheera_ai_params.mock_response and isinstance(dheera_ai_params.mock_response, str):

        return mock_response(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            mock_response=dheera_ai_params.mock_response,
        )

    anthropic_messages_provider_config: Optional[BaseAnthropicMessagesConfig] = None

    if custom_llm_provider is not None and custom_llm_provider in [
        provider.value for provider in LlmProviders
    ]:
        anthropic_messages_provider_config = (
            ProviderConfigManager.get_provider_anthropic_messages_config(
                model=model,
                provider=dheera_ai.LlmProviders(custom_llm_provider),
            )
        )
    if anthropic_messages_provider_config is None:
        # Handle non-Anthropic models using the adapter
        return (
            DheeraAIMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                metadata=metadata,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                _is_async=is_async,
                api_key=api_key,
                api_base=api_base,
                client=client,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )
        )

    if custom_llm_provider is None:
        raise ValueError(
            f"custom_llm_provider is required for Anthropic messages, passed in model={model}, custom_llm_provider={custom_llm_provider}"
        )

    local_vars.update(kwargs)
    anthropic_messages_optional_request_params = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params=local_vars
        )
    )
    return base_llm_http_handler.anthropic_messages_handler(
        model=model,
        messages=messages,
        anthropic_messages_provider_config=anthropic_messages_provider_config,
        anthropic_messages_optional_request_params=dict(
            anthropic_messages_optional_request_params
        ),
        _is_async=is_async,
        client=client,
        custom_llm_provider=custom_llm_provider,
        dheera_ai_params=dheera_ai_params,
        logging_obj=dheera_ai_logging_obj,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        kwargs=kwargs,
    )
