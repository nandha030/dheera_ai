"""
Responses API transformation for DheeraAI Proxy provider.

DheeraAI Proxy supports the OpenAI Responses API natively when the underlying model supports it.
This config enables pass-through behavior to the proxy's /v1/responses endpoint.
"""

from typing import Optional

from dheera_ai.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from dheera_ai.secret_managers.main import get_secret_str
from dheera_ai.types.utils import LlmProviders


class DheeraAIProxyResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for DheeraAI Proxy Responses API support.
    
    Extends OpenAI's config since the proxy follows OpenAI's API spec,
    but uses DHEERA_AI_PROXY_API_BASE for the base URL.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DHEERA_AI_PROXY

    def get_complete_url(
        self,
        api_base: Optional[str],
        dheera_ai_params: dict,
    ) -> str:
        """
        Get the endpoint for DheeraAI Proxy responses API.
        
        Uses DHEERA_AI_PROXY_API_BASE environment variable if api_base is not provided.
        """
        api_base = api_base or get_secret_str("DHEERA_AI_PROXY_API_BASE")
        
        if api_base is None:
            raise ValueError(
                "api_base not set for DheeraAI Proxy responses API. "
                "Set via api_base parameter or DHEERA_AI_PROXY_API_BASE environment variable"
            )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"
