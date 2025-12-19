from typing import Any, Dict, Optional, Tuple

from dheera_ai.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
from dheera_ai.llms.vertex_ai.vertex_llm_base import VertexBase


class VertexAITokenCounter(GoogleAIStudioTokenCounter, VertexBase):
    async def validate_environment(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        model: str = "",
        dheera_ai_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Returns a Tuple of headers and url for the Vertex AI countTokens endpoint.
        """
        dheera_ai_params = dheera_ai_params or {}
        vertex_credentials = self.get_vertex_ai_credentials(dheera_ai_params=dheera_ai_params)
        vertex_project = self.get_vertex_ai_project(dheera_ai_params=dheera_ai_params)
        vertex_location = self.get_vertex_ai_location(dheera_ai_params=dheera_ai_params)
        should_use_v1beta1_features = self.is_using_v1beta1_features(dheera_ai_params)
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=None,
            should_use_v1beta1_features=should_use_v1beta1_features,
            mode="count_tokens",
        )
        headers = {
            "Authorization": f"Bearer {auth_header}",
        }
        return headers, api_base