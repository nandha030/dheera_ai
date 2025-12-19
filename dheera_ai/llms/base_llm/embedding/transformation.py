from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from dheera_ai.llms.base_llm.chat.transformation import BaseConfig
from dheera_ai.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from dheera_ai.types.utils import EmbeddingResponse, ModelResponse

if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as _DheeraAILoggingObj

    DheeraAILoggingObj = _DheeraAILoggingObj
else:
    DheeraAILoggingObj = Any


class BaseEmbeddingConfig(BaseConfig, ABC):
    @abstractmethod
    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {}

    @abstractmethod
    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: DheeraAILoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        dheera_ai_params: dict,
    ) -> EmbeddingResponse:
        return model_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        dheera_ai_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        dheera_ai_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "EmbeddingConfig does not need a request transformation for chat models"
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: DheeraAILoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        dheera_ai_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "EmbeddingConfig does not need a response transformation for chat models"
        )
