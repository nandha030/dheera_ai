"""
Base transformation class for Interactions API implementations.

This follows the same pattern as BaseResponsesAPIConfig for the Responses API.

Per OpenAPI spec (https://ai.google.dev/static/api/interactions.openapi.json):
- Create: POST /{api_version}/interactions
- Get: GET /{api_version}/interactions/{interaction_id}
- Delete: DELETE /{api_version}/interactions/{interaction_id}
"""

import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from dheera_ai.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.types.utils import LlmProviders

if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as _DheeraAILoggingObj

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    DheeraAILoggingObj = _DheeraAILoggingObj
    BaseLLMException = _BaseLLMException
else:
    DheeraAILoggingObj = Any
    BaseLLMException = Any


class BaseInteractionsAPIConfig(ABC):
    """
    Base configuration class for Google Interactions API implementations.
    
    Per OpenAPI spec, the Interactions API supports two types of interactions:
    - Model interactions (with model parameter)
    - Agent interactions (with agent parameter)
    
    Implementations should override the abstract methods to provide
    provider-specific transformations for requests and responses.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        """Return the LLM provider identifier."""
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @abstractmethod
    def get_supported_params(self, model: str) -> List[str]:
        """
        Return the list of supported parameters for the given model.
        """
        pass

    @abstractmethod
    def validate_environment(
        self, 
        headers: dict, 
        model: str, 
        dheera_ai_params: Optional[GenericDheeraAIParams]
    ) -> dict:
        """
        Validate and prepare environment settings including headers.
        """
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        model: Optional[str],
        agent: Optional[str] = None,
        dheera_ai_params: Optional[dict] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the interaction request.
        
        Per OpenAPI spec: POST /{api_version}/interactions
        
        Args:
            api_base: Base URL for the API
            model: The model name (for model interactions)
            agent: The agent name (for agent interactions)
            dheera_ai_params: DheeraAI parameters
            stream: Whether this is a streaming request
            
        Returns:
            The complete URL for the request
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_request(
        self,
        model: Optional[str],
        agent: Optional[str],
        input: Optional[InteractionInput],
        optional_params: InteractionsAPIOptionalRequestParams,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Dict:
        """
        Transform the input request into the provider's expected format.
        
        Per OpenAPI spec, the request body should be either:
        - CreateModelInteractionParams (with model)
        - CreateAgentInteractionParams (with agent)
        
        Args:
            model: The model name (for model interactions)
            agent: The agent name (for agent interactions)
            input: The input content (string, content object, or list)
            optional_params: Optional parameters for the request
            dheera_ai_params: DheeraAI-specific parameters
            headers: Request headers
            
        Returns:
            The transformed request body as a dictionary
        """
        pass

    @abstractmethod
    def transform_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> InteractionsAPIResponse:
        """
        Transform the raw HTTP response into an InteractionsAPIResponse.
        
        Per OpenAPI spec, the response is an Interaction object.
        """
        pass

    @abstractmethod
    def transform_streaming_response(
        self,
        model: Optional[str],
        parsed_chunk: dict,
        logging_obj: DheeraAILoggingObj,
    ) -> InteractionsAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk into an InteractionsAPIStreamingResponse.
        
        Per OpenAPI spec, streaming uses SSE with various event types.
        """
        pass

    # =========================================================
    # GET INTERACTION TRANSFORMATION
    # =========================================================
    
    @abstractmethod
    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the get interaction request into URL and query params.
        
        Per OpenAPI spec: GET /{api_version}/interactions/{interaction_id}
        
        Returns:
            Tuple of (URL, query_params)
        """
        pass

    @abstractmethod
    def transform_get_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> InteractionsAPIResponse:
        """
        Transform the get interaction response.
        """
        pass

    # =========================================================
    # DELETE INTERACTION TRANSFORMATION
    # =========================================================
    
    @abstractmethod
    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the delete interaction request into URL and body.
        
        Per OpenAPI spec: DELETE /{api_version}/interactions/{interaction_id}
        
        Returns:
            Tuple of (URL, request_body)
        """
        pass

    @abstractmethod
    def transform_delete_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        interaction_id: str,
    ) -> DeleteInteractionResult:
        """
        Transform the delete interaction response.
        """
        pass

    # =========================================================
    # CANCEL INTERACTION TRANSFORMATION
    # =========================================================
    
    @abstractmethod
    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the cancel interaction request into URL and body.
        
        Returns:
            Tuple of (URL, request_body)
        """
        pass

    @abstractmethod
    def transform_cancel_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> CancelInteractionResult:
        """
        Transform the cancel interaction response.
        """
        pass

    # =========================================================
    # ERROR HANDLING
    # =========================================================

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate exception class for an error.
        """
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Returns True if dheera_ai should fake a stream for the given model.
        
        Override in subclasses if the provider doesn't support native streaming.
        """
        return False
