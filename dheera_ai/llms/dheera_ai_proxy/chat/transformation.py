"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

from dheera_ai.constants import OPENAI_CHAT_COMPLETION_PARAMS
from dheera_ai.secret_managers.main import get_secret_bool, get_secret_str
from dheera_ai.types.router import DheeraAI_Params

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from dheera_ai.types.llms.openai import AllMessageValues


class DheeraAIProxyChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List:
        params_list = super().get_supported_openai_params(model)
        params_list.extend(OPENAI_CHAT_COMPLETION_PARAMS)
        return params_list

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "thinking":
                optional_params.setdefault("extra_body", {})["thinking"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("DHEERA_AI_PROXY_API_BASE")  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DHEERA_AI_PROXY_API_KEY")
        return api_base, dynamic_api_key

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base, api_key = self._get_openai_compatible_provider_info(api_base, api_key)
        if api_base is None:
            raise ValueError(
                "api_base not set for DheeraAI Proxy route. Set in env via `DHEERA_AI_PROXY_API_BASE`"
            )
        models = super().get_models(api_key=api_key, api_base=api_base)
        return [f"dheera_ai_proxy/{model}" for model in models]

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("DHEERA_AI_PROXY_API_KEY")

    @staticmethod
    def _should_use_dheera_ai_proxy_by_default(
        dheera_ai_params: Optional[DheeraAI_Params] = None,
    ):
        """
        Returns True if dheera_ai proxy should be used by default for a given request

        Issue: https://github.com/BerriAI/dheera_ai/issues/10559

        Use case:
        - When using Google ADK, users want a flag to dynamically enable sending the request to dheera_ai proxy or not
        - Allow the model name to be passed in original format and still use dheera_ai proxy:
        "gemini/gemini-1.5-pro", "openai/gpt-4", "mistral/llama-2-70b-chat" etc.
        """
        import dheera_ai

        if get_secret_bool("USE_DHEERA_AI_PROXY") is True:
            return True
        if dheera_ai_params and dheera_ai_params.use_dheera_ai_proxy is True:
            return True
        if dheera_ai.use_dheera_ai_proxy is True:
            return True
        return False

    @staticmethod
    def dheera_ai_proxy_get_custom_llm_provider_info(
        model: str, api_base: Optional[str] = None, api_key: Optional[str] = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Force use dheera_ai proxy for all models

        Issue: https://github.com/BerriAI/dheera_ai/issues/10559

        Expected behavior:
        - custom_llm_provider will be 'dheera_ai_proxy'
        - api_base = api_base OR DHEERA_AI_PROXY_API_BASE
        - api_key = api_key OR DHEERA_AI_PROXY_API_KEY

        Use case:
        - When using Google ADK, users want a flag to dynamically enable sending the request to dheera_ai proxy or not
        -  Allow the model name to be passed in original format and still use dheera_ai proxy:
        "gemini/gemini-1.5-pro", "openai/gpt-4", "mistral/llama-2-70b-chat" etc.

        Return model, custom_llm_provider, dynamic_api_key, api_base
        """
        import dheera_ai

        custom_llm_provider = "dheera_ai_proxy"
        if model.startswith("dheera_ai_proxy/"):
            model = model.split("/", 1)[1]

        (
            api_base,
            api_key,
        ) = dheera_ai.DheeraAIProxyChatConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )

        return model, custom_llm_provider, api_key, api_base

    def transform_request(
        self,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        dheera_ai_params: dict,
        headers: dict,
    ) -> dict:
        # don't transform the request
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }

    async def async_transform_request(
        self,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        dheera_ai_params: dict,
        headers: dict,
    ) -> dict:
        # don't transform the request
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }
