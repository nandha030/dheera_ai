"""Abstraction function for OpenAI's realtime API"""

from typing import Any, Optional, cast

import dheera_ai
from dheera_ai import get_llm_provider
from dheera_ai.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from dheera_ai.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from dheera_ai.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from dheera_ai.secret_managers.main import get_secret_str
from dheera_ai.types.realtime import RealtimeQueryParams
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.types.utils import LlmProviders
from dheera_ai.utils import ProviderConfigManager

from ..dheera_ai_core_utils.get_dheera_ai_params import get_dheera_ai_params
from ..dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILogging
from ..llms.azure.realtime.handler import AzureOpenAIRealtime
from ..llms.openai.realtime.handler import OpenAIRealtime
from ..utils import client as wrapper_client
from ..llms.custom_httpx.http_handler import get_shared_realtime_ssl_context

azure_realtime = AzureOpenAIRealtime()
openai_realtime = OpenAIRealtime()
base_llm_http_handler = BaseLLMHTTPHandler()


@wrapper_client
async def _arealtime(
    model: str,
    websocket: Any,  # fastapi websocket
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    client: Optional[Any] = None,
    timeout: Optional[float] = None,
    query_params: Optional[RealtimeQueryParams] = None,
    **kwargs,
):
    """
    Private function to handle the realtime API call.

    For PROXY use only.
    """
    headers = cast(Optional[dict], kwargs.get("headers"))
    extra_headers = cast(Optional[dict], kwargs.get("extra_headers"))
    if headers is None:
        headers = {}
    if extra_headers is not None:
        headers.update(extra_headers)
    dheera_ai_logging_obj: DheeraAILogging = kwargs.get("dheera_ai_logging_obj")  # type: ignore
    user = kwargs.get("user", None)
    dheera_ai_params = GenericDheeraAIParams(**kwargs)

    dheera_ai_params_dict = get_dheera_ai_params(**kwargs)

    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    dheera_ai_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        dheera_ai_params=dheera_ai_params_dict,
        custom_llm_provider=_custom_llm_provider,
    )

    provider_config: Optional[BaseRealtimeConfig] = None
    if _custom_llm_provider in LlmProviders._member_map_.values():
        provider_config = ProviderConfigManager.get_provider_realtime_config(
            model=model,
            provider=LlmProviders(_custom_llm_provider),
        )
    if provider_config is not None:
        await base_llm_http_handler.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=dheera_ai_logging_obj,
            provider_config=provider_config,
            api_base=api_base,
            api_key=api_key,
            client=client,
            timeout=timeout,
            headers=headers,
        )
    elif _custom_llm_provider == "azure":
        api_base = (
            dynamic_api_base
            or dheera_ai_params.api_base
            or dheera_ai.api_base
            or get_secret_str("AZURE_API_BASE")
        )
        # set API KEY
        api_key = (
            dynamic_api_key
            or dheera_ai.api_key
            or dheera_ai.openai_key
            or get_secret_str("AZURE_API_KEY")
        )

        api_version = (
            api_version
            or dheera_ai_params.api_version
            or "2024-10-01-preview"
        )
        
        realtime_protocol = (
            kwargs.get("realtime_protocol")
            or "beta"
        )
        await azure_realtime.async_realtime(
            model=model,
            websocket=websocket,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=None,
            client=None,
            timeout=timeout,
            logging_obj=dheera_ai_logging_obj,
            realtime_protocol=realtime_protocol,
        )
    elif _custom_llm_provider == "openai":
        api_base = (
            dynamic_api_base
            or dheera_ai_params.api_base
            or dheera_ai.api_base
            or "https://api.openai.com/"
        )
        # set API KEY
        api_key = (
            dynamic_api_key
            or dheera_ai.api_key
            or dheera_ai.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

        await openai_realtime.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=dheera_ai_logging_obj,
            api_base=api_base,
            api_key=api_key,
            client=None,
            timeout=timeout,
            query_params=query_params,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


async def _realtime_health_check(
    model: str,
    custom_llm_provider: str,
    api_key: Optional[str],
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    realtime_protocol: Optional[str] = None,
):
    """
    Health check for realtime API - tries connection to the realtime API websocket

    Args:
        model: str - model name
        api_base: str - api base
        api_version: Optional[str] - api version
        api_key: str - api key
        custom_llm_provider: str - custom llm provider
        realtime_protocol: Optional[str] - protocol version ("GA"/"v1" for GA path, "beta"/None for beta path)

    Returns:
        bool - True if connection is successful, False otherwise
    Raises:
        Exception - if the connection is not successful
    """
    import websockets

    url: Optional[str] = None
    if custom_llm_provider == "azure":
        url = azure_realtime._construct_url(
            api_base=api_base or "",
            model=model,
            api_version=api_version or "2024-10-01-preview",
            realtime_protocol=realtime_protocol,
        )
    elif custom_llm_provider == "openai":
        url = openai_realtime._construct_url(
            api_base=api_base or "https://api.openai.com/", query_params={"model": model}
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    ssl_context = get_shared_realtime_ssl_context()
    async with websockets.connect(  # type: ignore
        url,
        extra_headers={
            "api-key": api_key,  # type: ignore
        },
        max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ssl=ssl_context,
    ):
        return True
