from typing import Optional

from dheera_ai.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from dheera_ai.secret_managers.main import get_secret_str


class DheeraAIProxyImageEditConfig(OpenAIImageEditConfig):
    """Configuration for image edit requests routed through DheeraAI Proxy."""

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        api_key = api_key or get_secret_str("DHEERA_AI_PROXY_API_KEY")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(
        self, model: str, api_base: Optional[str], dheera_ai_params: dict
    ) -> str:
        api_base = api_base or get_secret_str("DHEERA_AI_PROXY_API_BASE")
        if api_base is None:
            raise ValueError(
                "api_base not set for DheeraAI Proxy route. Set in env via `DHEERA_AI_PROXY_API_BASE`"
            )
        api_base = api_base.rstrip("/")
        return f"{api_base}/images/edits"
