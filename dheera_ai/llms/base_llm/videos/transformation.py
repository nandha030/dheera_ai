import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from dheera_ai.types.responses.main import *
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.types.videos.main import VideoCreateOptionalRequestParams

if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as _DheeraAILoggingObj
    from dheera_ai.types.videos.main import VideoObject as _VideoObject

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    DheeraAILoggingObj = _DheeraAILoggingObj
    BaseLLMException = _BaseLLMException
    VideoObject = _VideoObject
else:
    DheeraAILoggingObj = Any
    BaseLLMException = Any
    VideoObject = Any


class BaseVideoConfig(ABC):
    def __init__(self):
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
    def get_supported_openai_params(self, model: str) -> list:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        dheera_ai_params: Optional[GenericDheeraAIParams] = None,
    ) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        dheera_ai_params: dict,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        pass

    @abstractmethod
    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        pass

    @abstractmethod
    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video content request into a URL and data/params
        
        Returns:
            Tuple[str, Dict]: (url, params) for the video content request
        """
        pass

    @abstractmethod
    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> bytes:
        pass

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> bytes:
        """
        Async transform video content download response to bytes.
        Optional method - providers can override if they need async transformations
        (e.g., RunwayML for downloading video from CloudFront URL).
        
        Default implementation falls back to sync transform_video_content_response.
        
        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object
            
        Returns:
            Video content as bytes
        """
        # Default implementation: call sync version
        return self.transform_video_content_response(
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    @abstractmethod
    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video remix request into a URL and data
        
        Returns:
            Tuple[str, Dict]: (url, data) for the video remix request
        """
        pass

    @abstractmethod
    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        pass

    @abstractmethod
    def transform_video_list_request(
        self,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video list request into a URL and params
        
        Returns:
            Tuple[str, Dict]: (url, params) for the video list request
        """
        pass

    @abstractmethod
    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str,str]:
        pass

    @abstractmethod
    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video delete request into a URL and data
        
        Returns:
            Tuple[str, Dict]: (url, data) for the video delete request
        """
        pass

    @abstractmethod
    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> VideoObject:
        pass

    @abstractmethod
    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video retrieve request into a URL and data/params
        
        Returns:
            Tuple[str, Dict]: (url, params) for the video retrieve request
        """
        pass

    @abstractmethod
    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        pass

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
