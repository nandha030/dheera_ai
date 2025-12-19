import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx
from httpx._types import RequestFiles

from dheera_ai.types.images.main import ImageEditOptionalRequestParams
from dheera_ai.types.responses.main import *
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.types.utils import FileTypes

if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as _DheeraAILoggingObj
    from dheera_ai.utils import ImageResponse as _ImageResponse

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    DheeraAILoggingObj = _DheeraAILoggingObj
    BaseLLMException = _BaseLLMException
    ImageResponse = _ImageResponse
else:
    DheeraAILoggingObj = Any
    BaseLLMException = Any
    ImageResponse = Any


class BaseImageEditConfig(ABC):
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
        image_edit_optional_params: ImageEditOptionalRequestParams,
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
    def transform_image_edit_request(
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict,
        dheera_ai_params: GenericDheeraAIParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        pass

    @abstractmethod
    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
    ) -> ImageResponse:
        pass

    def use_multipart_form_data(self) -> bool:
        """
        Return True if the provider uses multipart/form-data for image edit requests.
        Return False if the provider uses JSON requests.

        Default is True for backwards compatibility with OpenAI-style providers.
        """
        return True

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
