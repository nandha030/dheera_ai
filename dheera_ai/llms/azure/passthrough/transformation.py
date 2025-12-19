from typing import TYPE_CHECKING, List, Optional, Tuple

import httpx

from dheera_ai.llms.azure.common_utils import BaseAzureLLM
from dheera_ai.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from dheera_ai.secret_managers.main import get_secret_str
from dheera_ai.types.llms.openai import AllMessageValues
from dheera_ai.types.router import GenericDheeraAIParams

if TYPE_CHECKING:
    from httpx import URL


class AzurePassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in request_data

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        dheera_ai_params: dict,
    ) -> Tuple["URL", str]:
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("Azure api base not found")

        dheera_ai_metadata = dheera_ai_params.get("dheera_ai_metadata") or {}
        model_group = dheera_ai_metadata.get("model_group")
        if model_group and model_group in endpoint:
            endpoint = endpoint.replace(model_group, model)

        complete_url = BaseAzureLLM._get_base_azure_url(
            api_base=base_target_url,
            dheera_ai_params=dheera_ai_params,
            route=endpoint,
            default_api_version=dheera_ai_params.get("api_version"),
        )
        return (
            httpx.URL(complete_url),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        dheera_ai_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers,
            dheera_ai_params=GenericDheeraAIParams(
                **{**dheera_ai_params, "api_key": api_key}
            ),
        )

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        return api_base or get_secret_str("AZURE_API_BASE")

    @staticmethod
    def get_api_key(
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        return api_key or get_secret_str("AZURE_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        return super().get_models(api_key, api_base)
