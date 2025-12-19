from typing import TYPE_CHECKING, Any, Dict, Optional

from dheera_ai.types.videos.main import VideoCreateOptionalRequestParams
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.llms.azure.common_utils import BaseAzureLLM
from dheera_ai.llms.openai.videos.transformation import OpenAIVideoConfig
if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as _DheeraAILoggingObj

    from ...base_llm.videos.transformation import BaseVideoConfig as _BaseVideoConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    DheeraAILoggingObj = _DheeraAILoggingObj
    BaseVideoConfig = _BaseVideoConfig
    BaseLLMException = _BaseLLMException
else:
    DheeraAILoggingObj = Any
    BaseVideoConfig = Any
    BaseLLMException = Any


class AzureVideoConfig(OpenAIVideoConfig):
    """
    Configuration class for OpenAI video generation.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for video generation.
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(video_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        dheera_ai_params: Optional[GenericDheeraAIParams] = None,
    ) -> dict:
        """
        Validate Azure environment and set up authentication headers.
        Uses _base_validate_azure_environment to properly handle credentials from dheera_ai_credential_name.
        """
        # If dheera_ai_params is provided, use it; otherwise create a new one
        if dheera_ai_params is None:
            dheera_ai_params = GenericDheeraAIParams()
        
        if api_key and not dheera_ai_params.api_key:
            dheera_ai_params.api_key = api_key
        
        # Use the base Azure validation method which properly handles:
        # 1. Credentials from dheera_ai_credential_name via dheera_ai_params
        # 2. Sets the correct "api-key" header (not "Authorization: Bearer")
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers, 
            dheera_ai_params=dheera_ai_params
        )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        dheera_ai_params: dict,
    ) -> str:
        """
        Constructs a complete URL for the API request.
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            dheera_ai_params=dheera_ai_params,
            route="/openai/v1/videos",
            default_api_version="",
        )