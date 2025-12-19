"""
Main OCR function for DheeraAI.
"""
import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import dheera_ai
from dheera_ai._logging import verbose_logger
from dheera_ai.constants import request_timeout
from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
from dheera_ai.llms.base_llm.ocr.transformation import BaseOCRConfig, OCRResponse
from dheera_ai.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@client
async def aocr(
    model: str,
    document: Dict[str, str],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> OCRResponse:
    """
    Async OCR function.
    
    Args:
        model: Model name (e.g., "mistral/mistral-ocr-latest")
        document: Document to process in Mistral format:
            {"type": "document_url", "document_url": "https://..."} for PDFs/docs or
            {"type": "image_url", "image_url": "https://..."} for images
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters (e.g., include_image_base64, pages, image_limit)
        
    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.
        
    Example:
        ```python
        import dheera_ai
        
        # OCR with PDF
        response = await dheera_ai.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            },
            include_image_base64=True
        )
        
        # OCR with image
        response = await dheera_ai.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": "https://example.com/image.png"
            }
        )
        
        # OCR with base64 encoded PDF
        response = await dheera_ai.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )
        ```
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aocr"] = True

        # Get custom llm provider
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = dheera_ai.get_llm_provider(
                model=model, api_base=api_base
            )

        func = partial(
            ocr,
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        if response is None:
            raise ValueError(
                f"Got an unexpected None response from the OCR API: {response}"
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
def ocr(
    model: str,
    document: Dict[str, str],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[OCRResponse, Coroutine[Any, Any, OCRResponse]]:
    """
    Synchronous OCR function.
    
    Args:
        model: Model name (e.g., "mistral/mistral-ocr-latest")
        document: Document to process in Mistral format:
            {"type": "document_url", "document_url": "https://..."} for PDFs/docs or
            {"type": "image_url", "image_url": "https://..."} for images
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters (e.g., include_image_base64, pages, image_limit)
        
    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.
        
    Example:
        ```python
        import dheera_ai
        
        # OCR with PDF
        response = dheera_ai.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            },
            include_image_base64=True
        )
        
        # OCR with image
        response = dheera_ai.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": "https://example.com/image.png"
            }
        )
        
        # OCR with base64 encoded PDF
        response = dheera_ai.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )
        
        # Access pages
        for page in response.pages:
            print(f"Page {page.index}: {page.markdown}")
        ```
    """
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.pop("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id", None)
        _is_async = kwargs.pop("aocr", False) is True
        
        # Validate document parameter format (Mistral spec)
        if not isinstance(document, dict):
            raise ValueError(f"document must be a dict with 'type' and URL field, got {type(document)}")
        
        doc_type = document.get("type")
        if doc_type not in ["document_url", "image_url"]:
            raise ValueError(f"Invalid document type: {doc_type}. Must be 'document_url' or 'image_url'")

        model, custom_llm_provider, dynamic_api_key, dynamic_api_base = (
            dheera_ai.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
            )
        )
        
        # Update with dynamic values if available
        if dynamic_api_key:
            api_key = dynamic_api_key
        if dynamic_api_base:
            api_base = dynamic_api_base

        # Get provider config
        ocr_provider_config: Optional[BaseOCRConfig] = (
            ProviderConfigManager.get_provider_ocr_config(
                model=model,
                provider=dheera_ai.LlmProviders(custom_llm_provider),
            )
        )

        if ocr_provider_config is None:
            raise ValueError(
                f"OCR is not supported for provider: {custom_llm_provider}"
            )

        verbose_logger.debug(
            f"OCR call - model: {model}, provider: {custom_llm_provider}"
        )

        # Get dheera_ai params using GenericDheeraAIParams (same as responses API)
        dheera_ai_params = GenericDheeraAIParams(**kwargs)
        
        # Extract OCR-specific parameters from kwargs
        supported_params = ocr_provider_config.get_supported_ocr_params(model=model)
        non_default_params = {}
        for param in supported_params:
            if param in kwargs:
                non_default_params[param] = kwargs.pop(param)
        
        # Map parameters to provider-specific format
        optional_params = ocr_provider_config.map_ocr_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
        )
        
        verbose_logger.debug(f"OCR optional_params after mapping: {optional_params}")

        # Pre Call logging
        dheera_ai_logging_obj.update_environment_variables(
            model=model,
            optional_params=optional_params,
            dheera_ai_params={
                "dheera_ai_call_id": dheera_ai_call_id,
                "api_base": api_base,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler - pass document dict directly
        response = base_llm_http_handler.ocr(
            model=model,
            document=document,  # Pass the entire document dict
            optional_params=optional_params,
            timeout=timeout or request_timeout,
            logging_obj=dheera_ai_logging_obj,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            aocr=_is_async,
            headers=extra_headers,
            provider_config=ocr_provider_config,
            dheera_ai_params=dict(dheera_ai_params),
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

