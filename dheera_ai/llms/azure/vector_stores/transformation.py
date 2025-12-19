from typing import Optional

from dheera_ai.llms.azure.common_utils import BaseAzureLLM
from dheera_ai.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from dheera_ai.types.router import GenericDheeraAIParams


class AzureOpenAIVectorStoreConfig(OpenAIVectorStoreConfig):
    def get_complete_url(
        self,
        api_base: Optional[str],
        dheera_ai_params: dict,
    ) -> str:
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            dheera_ai_params=dheera_ai_params,
            route="/openai/vector_stores"
        )


    def validate_environment(
        self, headers: dict,  dheera_ai_params: Optional[GenericDheeraAIParams]
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers,
            dheera_ai_params=dheera_ai_params
        )