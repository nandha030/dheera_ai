import asyncio
import contextvars
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Optional, Union

import httpx
from pydantic import BaseModel, ConfigDict

import dheera_ai
from dheera_ai.constants import request_timeout

# Import the adapter for fallback to completion format
from dheera_ai.google_genai.adapters.handler import GenerateContentToCompletionHandler
from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
from dheera_ai.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from dheera_ai.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.utils import ProviderConfigManager, client

if TYPE_CHECKING:
    from dheera_ai.types.google_genai.main import (
        GenerateContentConfigDict,
        GenerateContentContentListUnionDict,
        GenerateContentResponse,
        ToolConfigDict,
    )
else:
    GenerateContentConfigDict = Any
    GenerateContentContentListUnionDict = Any
    GenerateContentResponse = Any
    ToolConfigDict = Any


####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


class GenerateContentSetupResult(BaseModel):
    """Internal Type - Result of setting up a generate content call"""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    model: str
    request_body: Dict[str, Any]
    custom_llm_provider: str
    generate_content_provider_config: Optional[BaseGoogleGenAIGenerateContentConfig]
    generate_content_config_dict: Dict[str, Any]
    dheera_ai_params: GenericDheeraAIParams
    dheera_ai_logging_obj: DheeraAILoggingObj
    dheera_ai_call_id: Optional[str]


class GenerateContentHelper:
    """Helper class for Google GenAI generate content operations"""

    @staticmethod
    def mock_generate_content_response(
        mock_response: str = "This is a mock response from Google GenAI generate_content.",
    ) -> Dict[str, Any]:
        """Mock response for generate_content for testing purposes"""
        return {
            "text": mock_response,
            "candidates": [
                {
                    "content": {"parts": [{"text": mock_response}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        }

    @staticmethod
    def setup_generate_content_call(
        model: str,
        contents: GenerateContentContentListUnionDict,
        config: Optional[GenerateContentConfigDict] = None,
        custom_llm_provider: Optional[str] = None,
        tools: Optional[ToolConfigDict] = None,
        **kwargs,
    ) -> GenerateContentSetupResult:
        """
        Common setup logic for generate_content calls

        Args:
            model: The model name
            contents: The content to generate from
            config: Optional configuration
            custom_llm_provider: Optional custom LLM provider
            tools: Optional tools
            **kwargs: Additional keyword arguments

        Returns:
            GenerateContentSetupResult containing all setup information
        """
        dheera_ai_logging_obj: Optional[DheeraAILoggingObj] = kwargs.get(
            "dheera_ai_logging_obj"
        )
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id", None)

        # get llm provider logic
        dheera_ai_params = GenericDheeraAIParams(**kwargs)

        ## MOCK RESPONSE LOGIC (only for non-streaming)
        if (
            not kwargs.get("stream", False)
            and dheera_ai_params.mock_response
            and isinstance(dheera_ai_params.mock_response, str)
        ):
            raise ValueError("Mock response should be handled by caller")

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

        # get provider config
        generate_content_provider_config: Optional[
            BaseGoogleGenAIGenerateContentConfig
        ] = ProviderConfigManager.get_provider_google_genai_generate_content_config(
            model=model,
            provider=dheera_ai.LlmProviders(custom_llm_provider),
        )

        if generate_content_provider_config is None:
            # Use adapter to transform to completion format when provider config is None
            # Signal that we should use the adapter by returning special result
            if dheera_ai_logging_obj is None:
                raise ValueError("dheera_ai_logging_obj is required, but got None")
            return GenerateContentSetupResult(
                model=model,
                custom_llm_provider=custom_llm_provider,
                request_body={},  # Will be handled by adapter
                generate_content_provider_config=None,  # type: ignore
                generate_content_config_dict=dict(config or {}),
                dheera_ai_params=dheera_ai_params,
                dheera_ai_logging_obj=dheera_ai_logging_obj,
                dheera_ai_call_id=dheera_ai_call_id,
            )

        #########################################################################################
        # Construct request body
        #########################################################################################
        # Create Google Optional Params Config
        generate_content_config_dict = (
            generate_content_provider_config.map_generate_content_optional_params(
                generate_content_config_dict=config or {},
                model=model,
            )
        )
        # Extract systemInstruction from kwargs to pass to transform
        system_instruction = kwargs.get("systemInstruction") or kwargs.get("system_instruction")
        request_body = (
            generate_content_provider_config.transform_generate_content_request(
                model=model,
                contents=contents,
                tools=tools,
                generate_content_config_dict=generate_content_config_dict,
                system_instruction=system_instruction,
            )
        )

        # Pre Call logging
        if dheera_ai_logging_obj is None:
            raise ValueError("dheera_ai_logging_obj is required, but got None")

        dheera_ai_logging_obj.update_environment_variables(
            model=model,
            optional_params=dict(generate_content_config_dict),
            dheera_ai_params={
                "dheera_ai_call_id": dheera_ai_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        return GenerateContentSetupResult(
            model=model,
            custom_llm_provider=custom_llm_provider,
            request_body=request_body,
            generate_content_provider_config=generate_content_provider_config,
            generate_content_config_dict=generate_content_config_dict,
            dheera_ai_params=dheera_ai_params,
            dheera_ai_logging_obj=dheera_ai_logging_obj,
            dheera_ai_call_id=dheera_ai_call_id,
        )


@client
async def agenerate_content(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: Optional[GenerateContentConfigDict] = None,
    tools: Optional[ToolConfigDict] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # DheeraAI specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Async: Generate content using Google GenAI
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["agenerate_content"] = True

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = dheera_ai.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
            )

        func = partial(
            generate_content,
            model=model,
            contents=contents,
            config=config,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
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
    except Exception as e:
        raise dheera_ai.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: Optional[GenerateContentConfigDict] = None,
    tools: Optional[ToolConfigDict] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # DheeraAI specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Generate content using Google GenAI
    """
    local_vars = locals()
    try:
        _is_async = kwargs.pop("agenerate_content", False)

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # Check for mock response first
        dheera_ai_params = GenericDheeraAIParams(**kwargs)
        if dheera_ai_params.mock_response and isinstance(
            dheera_ai_params.mock_response, str
        ):
            return GenerateContentHelper.mock_generate_content_response(
                mock_response=dheera_ai_params.mock_response
            )

        # Setup the call
        setup_result = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        # Extract systemInstruction from kwargs to pass to handler
        system_instruction = kwargs.get("systemInstruction") or kwargs.get("system_instruction")

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            # Use the adapter to convert to completion format
            return GenerateContentToCompletionHandler.generate_content_handler(
                model=model,
                contents=contents,  # type: ignore
                config=setup_result.generate_content_config_dict,
                tools=tools,
                _is_async=_is_async,
                dheera_ai_params=setup_result.dheera_ai_params,
                **kwargs,
            )

        # Call the standard handler
        response = base_llm_http_handler.generate_content_handler(
            model=setup_result.model,
            contents=contents,
            tools=tools,
            generate_content_provider_config=setup_result.generate_content_provider_config,
            generate_content_config_dict=setup_result.generate_content_config_dict,
            custom_llm_provider=setup_result.custom_llm_provider,
            dheera_ai_params=setup_result.dheera_ai_params,
            logging_obj=setup_result.dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            dheera_ai_metadata=kwargs.get("dheera_ai_metadata", {}),
            system_instruction=system_instruction,
        )

        return response
    except Exception as e:
        raise dheera_ai.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def agenerate_content_stream(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: Optional[GenerateContentConfigDict] = None,
    tools: Optional[ToolConfigDict] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # DheeraAI specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Async: Generate content using Google GenAI with streaming response
    """
    local_vars = locals()
    try:
        kwargs["agenerate_content_stream"] = True

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = dheera_ai.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        # Setup the call
        setup_result = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        # Extract systemInstruction from kwargs to pass to handler
        system_instruction = kwargs.get("systemInstruction") or kwargs.get("system_instruction")

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            # Use the adapter to convert to completion format
            return (
                await GenerateContentToCompletionHandler.async_generate_content_handler(
                    model=model,
                    contents=contents,  # type: ignore
                    config=setup_result.generate_content_config_dict,
                    dheera_ai_params=setup_result.dheera_ai_params,
                    tools=tools,
                    stream=True,
                    **kwargs,
                )
            )

        # Call the handler with async enabled and streaming
        # Return the coroutine directly for the router to handle
        return await base_llm_http_handler.generate_content_handler(
            model=setup_result.model,
            contents=contents,
            generate_content_provider_config=setup_result.generate_content_provider_config,
            generate_content_config_dict=setup_result.generate_content_config_dict,
            tools=tools,
            custom_llm_provider=setup_result.custom_llm_provider,
            dheera_ai_params=setup_result.dheera_ai_params,
            logging_obj=setup_result.dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=True,
            client=kwargs.get("client"),
            stream=True,
            dheera_ai_metadata=kwargs.get("dheera_ai_metadata", {}),
            system_instruction=system_instruction,
        )

    except Exception as e:
        raise dheera_ai.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content_stream(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: Optional[GenerateContentConfigDict] = None,
    tools: Optional[ToolConfigDict] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # DheeraAI specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """
    Generate content using Google GenAI with streaming response
    """
    local_vars = locals()
    try:
        # Remove any async-related flags since this is the sync function
        _is_async = kwargs.pop("agenerate_content_stream", False)

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # Setup the call
        setup_result = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            # Use the adapter to convert to completion format
            return GenerateContentToCompletionHandler.generate_content_handler(
                model=model,
                contents=contents,  # type: ignore
                config=setup_result.generate_content_config_dict,
                _is_async=_is_async,
                dheera_ai_params=setup_result.dheera_ai_params,
                stream=True,
                **kwargs,
            )

        # Call the handler with streaming enabled (sync version)
        return base_llm_http_handler.generate_content_handler(
            model=setup_result.model,
            contents=contents,
            generate_content_provider_config=setup_result.generate_content_provider_config,
            generate_content_config_dict=setup_result.generate_content_config_dict,
            tools=tools,
            custom_llm_provider=setup_result.custom_llm_provider,
            dheera_ai_params=setup_result.dheera_ai_params,
            logging_obj=setup_result.dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            stream=True,
            dheera_ai_metadata=kwargs.get("dheera_ai_metadata", {}),
        )

    except Exception as e:
        raise dheera_ai.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
