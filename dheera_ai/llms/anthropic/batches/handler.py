"""
Anthropic Batches API Handler
"""

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Union

import httpx

from dheera_ai.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
)
from dheera_ai.types.utils import DheeraAIBatch, LlmProviders

if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
else:
    DheeraAILoggingObj = Any

from ..common_utils import AnthropicModelInfo
from .transformation import AnthropicBatchesConfig


class AnthropicBatchesHandler:
    """
    Handler for Anthropic Message Batches API.
    
    Supports:
    - retrieve_batch() - Retrieve batch status and information
    """

    def __init__(self):
        self.anthropic_model_info = AnthropicModelInfo()
        self.provider_config = AnthropicBatchesConfig()

    async def aretrieve_batch(
        self,
        batch_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        logging_obj: Optional[DheeraAILoggingObj] = None,
    ) -> DheeraAIBatch:
        """
        Async: Retrieve a batch from Anthropic.
        
        Args:
            batch_id: The batch ID to retrieve
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            logging_obj: Optional logging object
            
        Returns:
            DheeraAIBatch: Batch information in OpenAI format
        """
        # Resolve API credentials
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        api_key = api_key or self.anthropic_model_info.get_api_key()
        
        if not api_key:
            raise ValueError("Missing Anthropic API Key")
        
        # Create a minimal logging object if not provided
        if logging_obj is None:
            from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObjClass
            logging_obj = DheeraAILoggingObjClass(
                model="anthropic/unknown",
                messages=[],
                stream=False,
                call_type="batch_retrieve",
                start_time=None,
                dheera_ai_call_id=f"batch_retrieve_{batch_id}",
                function_id="batch_retrieve",
            )
        
        # Get the complete URL for batch retrieval
        retrieve_url = self.provider_config.get_retrieve_batch_url(
            api_base=api_base,
            batch_id=batch_id,
            optional_params={},
            dheera_ai_params={},
        )
        
        # Validate environment and get headers
        headers = self.provider_config.validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            dheera_ai_params={},
            api_key=api_key,
            api_base=api_base,
        )

        logging_obj.pre_call(
            input=batch_id,
            api_key=api_key,
            additional_args={
                "api_base": retrieve_url,
                "headers": headers,
                "complete_input_dict": {},
            },
        )
        # Make the request
        async_client = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC)
        response = await async_client.get(
            url=retrieve_url,
            headers=headers
        )
        response.raise_for_status()
        
        # Transform response to DheeraAI format
        return self.provider_config.transform_retrieve_batch_response(
            model=None,
            raw_response=response,
            logging_obj=logging_obj,
            dheera_ai_params={},
        )

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        logging_obj: Optional[DheeraAILoggingObj] = None,
    ) -> Union[DheeraAIBatch, Coroutine[Any, Any, DheeraAIBatch]]:
        """
        Retrieve a batch from Anthropic.
        
        Args:
            _is_async: Whether to run asynchronously
            batch_id: The batch ID to retrieve
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            logging_obj: Optional logging object
            
        Returns:
            DheeraAIBatch or Coroutine: Batch information in OpenAI format
        """
        if _is_async:
            return self.aretrieve_batch(
                batch_id=batch_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        else:
            return asyncio.run(
                self.aretrieve_batch(
                    batch_id=batch_id,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    logging_obj=logging_obj,
                )
            )

