from typing import Optional, Tuple

import httpx

import dheera_ai
from dheera_ai.constants import REPLICATE_MODEL_NAME_WITH_ID_LENGTH
from dheera_ai.secret_managers.main import get_secret, get_secret_str

from ..types.router import DheeraAI_Params


def _is_non_openai_azure_model(model: str) -> bool:
    try:
        model_name = model.split("/", 1)[1]
        if (
            model_name in dheera_ai.cohere_chat_models
            or f"mistral/{model_name}" in dheera_ai.mistral_chat_models
        ):
            return True
    except Exception:
        return False
    return False


def _is_azure_claude_model(model: str) -> bool:
    """
    Check if a model name contains 'claude' (case-insensitive).
    Used to detect Claude models that need Anthropic-specific handling.
    """
    try:
        model_lower = model.lower()
        return "claude" in model_lower or model_lower.startswith("claude")
    except Exception:
        return False


def handle_cohere_chat_model_custom_llm_provider(
    model: str, custom_llm_provider: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    if user sets model = "cohere/command-r" -> use custom_llm_provider = "cohere_chat"

    Args:
        model:
        custom_llm_provider:

    Returns:
        model, custom_llm_provider
    """

    if custom_llm_provider:
        if custom_llm_provider == "cohere" and model in dheera_ai.cohere_chat_models:
            return model, "cohere_chat"

    if "/" in model:
        _custom_llm_provider, _model = model.split("/", 1)
        if (
            _custom_llm_provider
            and _custom_llm_provider == "cohere"
            and _model in dheera_ai.cohere_chat_models
        ):
            return _model, "cohere_chat"

    return model, custom_llm_provider


def handle_anthropic_text_model_custom_llm_provider(
    model: str, custom_llm_provider: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    if user sets model = "anthropic/claude-2" -> use custom_llm_provider = "anthropic_text"

    Args:
        model:
        custom_llm_provider:

    Returns:
        model, custom_llm_provider
    """

    if custom_llm_provider:
        if (
            custom_llm_provider == "anthropic"
            and dheera_ai.AnthropicTextConfig._is_anthropic_text_model(model)
        ):
            return model, "anthropic_text"

    if "/" in model:
        _custom_llm_provider, _model = model.split("/", 1)
        if (
            _custom_llm_provider
            and _custom_llm_provider == "anthropic"
            and dheera_ai.AnthropicTextConfig._is_anthropic_text_model(_model)
        ):
            return _model, "anthropic_text"

    return model, custom_llm_provider


def get_llm_provider(  # noqa: PLR0915
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    dheera_ai_params: Optional[DheeraAI_Params] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns the provider for a given model name - e.g. 'azure/chatgpt-v-2' -> 'azure'

    For router -> Can also give the whole dheera_ai param dict -> this function will extract the relevant details

    Raises Error - if unable to map model to a provider

    Return model, custom_llm_provider, dynamic_api_key, api_base
    """
    try:
        if dheera_ai.DheeraAIProxyChatConfig._should_use_dheera_ai_proxy_by_default(
            dheera_ai_params=dheera_ai_params
        ):
            return dheera_ai.DheeraAIProxyChatConfig.dheera_ai_proxy_get_custom_llm_provider_info(
                model=model, api_base=api_base, api_key=api_key
            )

        ## IF LITELLM PARAMS GIVEN ##
        if dheera_ai_params:
            assert (
                custom_llm_provider is None and api_base is None and api_key is None
            ), "Either pass in dheera_ai_params or the custom_llm_provider/api_base/api_key. Otherwise, these values will be overriden."
            custom_llm_provider = dheera_ai_params.custom_llm_provider
            api_base = dheera_ai_params.api_base
            api_key = dheera_ai_params.api_key

        dynamic_api_key = None
        # check if llm provider provided
        # AZURE AI-Studio Logic - Azure AI Studio supports AZURE/Cohere
        # If User passes azure/command-r-plus -> we should send it to cohere_chat/command-r-plus
        if model.split("/", 1)[0] == "azure":
            if _is_non_openai_azure_model(model):
                custom_llm_provider = "openai"
                return model, custom_llm_provider, dynamic_api_key, api_base

        ### Handle cases when custom_llm_provider is set to cohere/command-r-plus but it should use cohere_chat route
        model, custom_llm_provider = handle_cohere_chat_model_custom_llm_provider(
            model, custom_llm_provider
        )

        model, custom_llm_provider = handle_anthropic_text_model_custom_llm_provider(
            model, custom_llm_provider
        )

        if custom_llm_provider and (
            model.split("/")[0] != custom_llm_provider
        ):  # handle scenario where model="azure/*" and custom_llm_provider="azure"
            model = custom_llm_provider + "/" + model

        if api_key and api_key.startswith("os.environ/"):
            dynamic_api_key = get_secret_str(api_key)
        # check if llm provider part of model name

        if (
            model.split("/", 1)[0] in dheera_ai.provider_list
            and model.split("/", 1)[0] not in dheera_ai.model_list_set
            and len(model.split("/"))
            > 1  # handle edge case where user passes in `dheera_ai --model mistral` https://github.com/BerriAI/dheera_ai/issues/1351
        ):
            return _get_openai_compatible_provider_info(
                model=model,
                api_base=api_base,
                api_key=api_key,
                dynamic_api_key=dynamic_api_key,
            )
        elif model.split("/", 1)[0] in dheera_ai.provider_list:
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(api_base)
                )
            if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        # check if api base is a known openai compatible endpoint
        if api_base:
            for endpoint in dheera_ai.openai_compatible_endpoints:
                if endpoint in api_base:
                    if endpoint == "api.perplexity.ai":
                        custom_llm_provider = "perplexity"
                        dynamic_api_key = get_secret_str("PERPLEXITYAI_API_KEY")
                    elif endpoint == "api.endpoints.anyscale.com/v1":
                        custom_llm_provider = "anyscale"
                        dynamic_api_key = get_secret_str("ANYSCALE_API_KEY")
                    elif endpoint == "api.deepinfra.com/v1/openai":
                        custom_llm_provider = "deepinfra"
                        dynamic_api_key = get_secret_str("DEEPINFRA_API_KEY")
                    elif endpoint == "api.mistral.ai/v1":
                        custom_llm_provider = "mistral"
                        dynamic_api_key = get_secret_str("MISTRAL_API_KEY")
                    elif endpoint == "api.groq.com/openai/v1":
                        custom_llm_provider = "groq"
                        dynamic_api_key = get_secret_str("GROQ_API_KEY")
                    elif endpoint == "https://integrate.api.nvidia.com/v1":
                        custom_llm_provider = "nvidia_nim"
                        dynamic_api_key = get_secret_str("NVIDIA_NIM_API_KEY")
                    elif endpoint == "https://api.cerebras.ai/v1":
                        custom_llm_provider = "cerebras"
                        dynamic_api_key = get_secret_str("CEREBRAS_API_KEY")
                    elif endpoint == "https://inference.baseten.co/v1":
                        custom_llm_provider = "baseten"
                        dynamic_api_key = get_secret_str("BASETEN_API_KEY")
                    elif endpoint == "https://api.sambanova.ai/v1":
                        custom_llm_provider = "sambanova"
                        dynamic_api_key = get_secret_str("SAMBANOVA_API_KEY")
                    elif endpoint == "https://api.ai21.com/studio/v1":
                        custom_llm_provider = "ai21_chat"
                        dynamic_api_key = get_secret_str("AI21_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "codestral"
                        dynamic_api_key = get_secret_str("CODESTRAL_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "text-completion-codestral"
                        dynamic_api_key = get_secret_str("CODESTRAL_API_KEY")
                    elif endpoint == "app.empower.dev/api/v1":
                        custom_llm_provider = "empower"
                        dynamic_api_key = get_secret_str("EMPOWER_API_KEY")
                    elif endpoint == "api.deepseek.com/v1":
                        custom_llm_provider = "deepseek"
                        dynamic_api_key = get_secret_str("DEEPSEEK_API_KEY")
                    elif endpoint == "ollama.com":
                        custom_llm_provider = "ollama"
                        dynamic_api_key = get_secret_str("OLLAMA_API_KEY")
                    elif endpoint == "https://api.friendli.ai/serverless/v1":
                        custom_llm_provider = "friendliai"
                        dynamic_api_key = get_secret_str(
                            "FRIENDLIAI_API_KEY"
                        ) or get_secret("FRIENDLI_TOKEN")
                    elif endpoint == "api.galadriel.com/v1":
                        custom_llm_provider = "galadriel"
                        dynamic_api_key = get_secret_str("GALADRIEL_API_KEY")
                    elif endpoint == "https://api.llama.com/compat/v1":
                        custom_llm_provider = "meta_llama"
                        dynamic_api_key = api_key or get_secret_str("LLAMA_API_KEY")
                    elif endpoint == "https://api.featherless.ai/v1":
                        custom_llm_provider = "featherless_ai"
                        dynamic_api_key = get_secret_str("FEATHERLESS_AI_API_KEY")
                    elif endpoint == dheera_ai.NscaleConfig.API_BASE_URL:
                        custom_llm_provider = "nscale"
                        dynamic_api_key = dheera_ai.NscaleConfig.get_api_key()
                    elif endpoint == "dashscope-intl.aliyuncs.com/compatible-mode/v1":
                        custom_llm_provider = "dashscope"
                        dynamic_api_key = get_secret_str("DASHSCOPE_API_KEY")
                    elif endpoint == "api.moonshot.ai/v1":
                        custom_llm_provider = "moonshot"
                        dynamic_api_key = get_secret_str("MOONSHOT_API_KEY")
                    elif endpoint == "platform.publicai.co/v1":
                        custom_llm_provider = "publicai"
                        dynamic_api_key = get_secret_str("PUBLICAI_API_KEY")
                    elif endpoint == "https://api.v0.dev/v1":
                        custom_llm_provider = "v0"
                        dynamic_api_key = get_secret_str("V0_API_KEY")
                    elif endpoint == "https://api.lambda.ai/v1":
                        custom_llm_provider = "lambda_ai"
                        dynamic_api_key = get_secret_str("LAMBDA_API_KEY")
                    elif endpoint == "https://api.hyperbolic.xyz/v1":
                        custom_llm_provider = "hyperbolic"
                        dynamic_api_key = get_secret_str("HYPERBOLIC_API_KEY")
                    elif endpoint == "https://ai-gateway.vercel.sh/v1":
                        custom_llm_provider = "vercel_ai_gateway"
                        dynamic_api_key = get_secret_str("VERCEL_AI_GATEWAY_API_KEY")
                    elif endpoint == "https://api.inference.wandb.ai/v1":
                        custom_llm_provider = "wandb"
                        dynamic_api_key = get_secret_str("WANDB_API_KEY")

                    if api_base is not None and not isinstance(api_base, str):
                        raise Exception(
                            "api base needs to be a string. api_base={}".format(
                                api_base
                            )
                        )
                    if dynamic_api_key is not None and not isinstance(
                        dynamic_api_key, str
                    ):
                        raise Exception(
                            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                                dynamic_api_key
                            )
                        )
                    return model, custom_llm_provider, dynamic_api_key, api_base  # type: ignore

        # check if model in known model provider list  -> for huggingface models, raise exception as they don't have a fixed provider (can be togetherai, anyscale, baseten, runpod, et.)
        ## openai - chatcompletion + text completion
        if (
            model in dheera_ai.open_ai_chat_completion_models
            or "ft:gpt-3.5-turbo" in model
            or "ft:gpt-4" in model  # catches ft:gpt-4-0613, ft:gpt-4o
            or model in dheera_ai.openai_image_generation_models
            or model in dheera_ai.openai_video_generation_models
        ):
            custom_llm_provider = "openai"
        elif model in dheera_ai.open_ai_text_completion_models:
            custom_llm_provider = "text-completion-openai"
        ## anthropic
        elif model in dheera_ai.anthropic_models:
            if dheera_ai.AnthropicTextConfig._is_anthropic_text_model(model):
                custom_llm_provider = "anthropic_text"
            else:
                custom_llm_provider = "anthropic"
        ## cohere
        elif model in dheera_ai.cohere_models or model in dheera_ai.cohere_embedding_models:
            custom_llm_provider = "cohere"
        ## cohere chat models
        elif model in dheera_ai.cohere_chat_models:
            custom_llm_provider = "cohere_chat"
        ## replicate
        elif model in dheera_ai.replicate_models or (
            ":" in model and len(model) > REPLICATE_MODEL_NAME_WITH_ID_LENGTH
        ):
            model_parts = model.split(":")
            if (
                len(model_parts) > 1
                and len(model_parts[1]) == REPLICATE_MODEL_NAME_WITH_ID_LENGTH
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                custom_llm_provider = "replicate"
            elif model in dheera_ai.replicate_models:
                custom_llm_provider = "replicate"
        ## openrouter
        elif model in dheera_ai.openrouter_models:
            custom_llm_provider = "openrouter"
        ## maritalk
        elif model in dheera_ai.maritalk_models:
            custom_llm_provider = "maritalk"
        ## vertex - text + chat + language (gemini) models
        elif (
            model in dheera_ai.vertex_chat_models
            or model in dheera_ai.vertex_code_chat_models
            or model in dheera_ai.vertex_text_models
            or model in dheera_ai.vertex_code_text_models
            or model in dheera_ai.vertex_language_models
            or model in dheera_ai.vertex_embedding_models
            or model in dheera_ai.vertex_vision_models
            or model in dheera_ai.vertex_ai_image_models
            or model in dheera_ai.vertex_ai_video_models
        ):
            custom_llm_provider = "vertex_ai"
        ## ai21
        elif model in dheera_ai.ai21_chat_models or model in dheera_ai.ai21_models:
            custom_llm_provider = "ai21_chat"
            api_base = (
                api_base
                or get_secret("AI21_API_BASE")
                or "https://api.ai21.com/studio/v1"
            )  # type: ignore
            dynamic_api_key = api_key or get_secret("AI21_API_KEY")
        ## aleph_alpha
        elif model in dheera_ai.aleph_alpha_models:
            custom_llm_provider = "aleph_alpha"
        ## baseten
        elif model in dheera_ai.baseten_models:
            custom_llm_provider = "baseten"
        ## nlp_cloud
        elif model in dheera_ai.nlp_cloud_models:
            custom_llm_provider = "nlp_cloud"
        ## petals
        elif model in dheera_ai.petals_models:
            custom_llm_provider = "petals"
        ## bedrock
        elif (
            model in dheera_ai.bedrock_models
            or model in dheera_ai.bedrock_embedding_models
            or model in dheera_ai.bedrock_converse_models
        ):
            custom_llm_provider = "bedrock"
        elif model in dheera_ai.watsonx_models:
            custom_llm_provider = "watsonx"
        # openai embeddings
        elif model in dheera_ai.open_ai_embedding_models:
            custom_llm_provider = "openai"
        elif model in dheera_ai.empower_models:
            custom_llm_provider = "empower"
        elif model in dheera_ai.gradient_ai_models:
            custom_llm_provider = "gradient_ai"
        elif model == "*":
            custom_llm_provider = "openai"
        # bytez models
        elif model.startswith("bytez/"):
            custom_llm_provider = "bytez"
        elif model.startswith("lemonade/"):
            custom_llm_provider = "lemonade"
        elif model.startswith("heroku/"):
            custom_llm_provider = "heroku"
        # cometapi models
        elif model.startswith("cometapi/"):
            custom_llm_provider = "cometapi"
        elif model.startswith("oci/"):
            custom_llm_provider = "oci"
        elif model.startswith("compactifai/"):
            custom_llm_provider = "compactifai"
        elif model.startswith("ovhcloud/"):
            custom_llm_provider = "ovhcloud"
        elif model.startswith("lemonade/"):
            custom_llm_provider = "lemonade"
        elif model.startswith("clarifai/"):
            custom_llm_provider = "clarifai"
        elif model.startswith("amazon_nova"):
            custom_llm_provider = "amazon_nova"
        elif model.startswith("sap/"):
            custom_llm_provider = "sap"
        if not custom_llm_provider:
            if dheera_ai.suppress_debug_info is False:
                print()  # noqa
                print(  # noqa
                    "\033[1;31mProvider List: https://docs.dheera_ai.ai/docs/providers\033[0m"  # noqa
                )  # noqa
                print()  # noqa
            error_str = f"LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model={model}\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.dheera_ai.ai/docs/providers"
            # maps to openai.NotFoundError, this is raised when openai does not recognize the llm
            raise dheera_ai.exceptions.BadRequestError(  # type: ignore
                message=error_str,
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/dheera_ai"),  # type: ignore
                ),
                llm_provider="",
            )
        if api_base is not None and not isinstance(api_base, str):
            raise Exception(
                "api base needs to be a string. api_base={}".format(api_base)
            )
        if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
            raise Exception(
                "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                    dynamic_api_key
                )
            )
        return model, custom_llm_provider, dynamic_api_key, api_base
    except Exception as e:
        if isinstance(e, dheera_ai.exceptions.BadRequestError):
            raise e
        else:
            error_str = (
                f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}"
            )
            raise dheera_ai.exceptions.BadRequestError(  # type: ignore
                message=f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}",
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/dheera_ai"),  # type: ignore
                ),
                llm_provider="",
            )


def _get_openai_compatible_provider_info(  # noqa: PLR0915
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    dynamic_api_key: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns:
        Tuple[str, str, Optional[str], Optional[str]]:
            model: str
            custom_llm_provider: str
            dynamic_api_key: Optional[str]
            api_base: Optional[str]
    """

    custom_llm_provider = model.split("/", 1)[0]
    model = model.split("/", 1)[1]

    # Check JSON providers FIRST (before hardcoded ones)
    from dheera_ai.llms.openai_like.dynamic_config import create_config_class
    from dheera_ai.llms.openai_like.json_loader import JSONProviderRegistry

    if JSONProviderRegistry.exists(custom_llm_provider):
        provider_config = JSONProviderRegistry.get(custom_llm_provider)
        if provider_config is None:
            raise ValueError(f"Provider {custom_llm_provider} not found")
        config_class = create_config_class(provider_config)
        api_base, dynamic_api_key = config_class()._get_openai_compatible_provider_info(
            api_base, api_key
        )
        return model, custom_llm_provider, dynamic_api_key, api_base

    if custom_llm_provider == "perplexity":
        # perplexity is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.perplexity.ai
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.PerplexityChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "aiohttp_openai":
        return model, "aiohttp_openai", api_key, api_base
    elif custom_llm_provider == "anyscale":
        # anyscale is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = api_base or get_secret_str("ANYSCALE_API_BASE") or "https://api.endpoints.anyscale.com/v1"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("ANYSCALE_API_KEY")
    elif custom_llm_provider == "deepinfra":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.DeepInfraConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "empower":
        api_base = (
            api_base
            or get_secret("EMPOWER_API_BASE")
            or "https://app.empower.dev/api/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("EMPOWER_API_KEY")
    elif custom_llm_provider == "groq":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.GroqChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "nvidia_nim":
        # nvidia_nim is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret("NVIDIA_NIM_API_BASE")
            or "https://integrate.api.nvidia.com/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NVIDIA_NIM_API_KEY")
    elif custom_llm_provider == "cerebras":
        api_base = (
            api_base or get_secret("CEREBRAS_API_BASE") or "https://api.cerebras.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CEREBRAS_API_KEY")
    elif custom_llm_provider == "baseten":
        # Use BasetenConfig to determine the appropriate API base URL
        if api_base is None:
            api_base = dheera_ai.BasetenConfig.get_api_base_for_model(model)
        else:
            api_base = api_base or get_secret_str("BASETEN_API_BASE") or "https://inference.baseten.co/v1"
        dynamic_api_key = api_key or get_secret_str("BASETEN_API_KEY")
    elif custom_llm_provider == "sambanova":
        api_base = (
            api_base
            or get_secret("SAMBANOVA_API_BASE")
            or "https://api.sambanova.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("SAMBANOVA_API_KEY")
    elif custom_llm_provider == "meta_llama":
        api_base = (
            api_base
            or get_secret("LLAMA_API_BASE")
            or "https://api.llama.com/compat/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("LLAMA_API_KEY")
    elif custom_llm_provider == "nebius":
        api_base = (
            api_base
            or get_secret("NEBIUS_API_BASE")
            or "https://api.studio.nebius.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NEBIUS_API_KEY")
    elif custom_llm_provider == "ollama":
        api_base = (
            api_base
            or get_secret("OLLAMA_API_BASE")
            or "http://localhost:11434"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("OLLAMA_API_KEY")
    elif (custom_llm_provider == "ai21_chat") or (
        custom_llm_provider == "ai21" and model in dheera_ai.ai21_chat_models
    ):
        api_base = (
            api_base or get_secret("AI21_API_BASE") or "https://api.ai21.com/studio/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("AI21_API_KEY")
        custom_llm_provider = "ai21_chat"
    elif custom_llm_provider == "volcengine":
        # volcengine is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret("VOLCENGINE_API_BASE")
            or "https://ark.cn-beijing.volces.com/api/v3"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("VOLCENGINE_API_KEY")
    elif custom_llm_provider == "codestral":
        # codestral is openai compatible, we just need to set this to custom_openai and have the api_base be https://codestral.mistral.ai/v1
        api_base = (
            api_base
            or get_secret("CODESTRAL_API_BASE")
            or "https://codestral.mistral.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CODESTRAL_API_KEY")
    elif custom_llm_provider == "hosted_vllm":
        # vllm is openai compatible, we just need to set this to custom_openai
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.HostedVLLMChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "llamafile":
        # llamafile is OpenAI compatible.
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.LlamafileChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "datarobot":
        # DataRobot is OpenAI compatible.
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.DataRobotConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "lm_studio":
        # lm_studio is openai compatible, we just need to set this to custom_openai
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.LMStudioChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "deepseek":
        # deepseek is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.deepseek.com/v1
        api_base = (
            api_base
            or get_secret("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore

        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
    elif custom_llm_provider == "fireworks_ai":
        # fireworks is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.fireworks.ai/inference/v1
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.FireworksAIConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
    elif custom_llm_provider == "azure_ai":
        (
            api_base,
            dynamic_api_key,
            custom_llm_provider,
        ) = dheera_ai.AzureAIStudioConfig()._get_openai_compatible_provider_info(
            model, api_base, api_key, custom_llm_provider
        )
    elif custom_llm_provider == "github":
        api_base = (
            api_base
            or get_secret_str("GITHUB_API_BASE")
            or "https://models.inference.ai.azure.com"  # This is github's default base url
        )
        dynamic_api_key = api_key or get_secret_str("GITHUB_API_KEY")
    elif custom_llm_provider == "dheera_ai_proxy":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.DheeraAIProxyChatConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )

    elif custom_llm_provider == "mistral":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.MistralConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "jina_ai":
        (
            custom_llm_provider,
            api_base,
            dynamic_api_key,
        ) = dheera_ai.JinaAIEmbeddingConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "xai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.XAIChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "zai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.ZAIChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "together_ai":
        api_base = (
            api_base
            or get_secret_str("TOGETHER_AI_API_BASE")
            or "https://api.together.xyz/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("TOGETHER_API_KEY")
            or get_secret_str("TOGETHER_AI_API_KEY")
            or get_secret_str("TOGETHERAI_API_KEY")
            or get_secret_str("TOGETHER_AI_TOKEN")
        )
    elif custom_llm_provider == "friendliai":
        api_base = (
            api_base
            or get_secret("FRIENDLI_API_BASE")
            or "https://api.friendli.ai/serverless/v1"
        )  # type: ignore
        dynamic_api_key = (
            api_key
            or get_secret_str("FRIENDLIAI_API_KEY")
            or get_secret_str("FRIENDLI_TOKEN")
        )
    elif custom_llm_provider == "galadriel":
        api_base = (
            api_base
            or get_secret("GALADRIEL_API_BASE")
            or "https://api.galadriel.com/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("GALADRIEL_API_KEY")
    elif custom_llm_provider == "github_copilot":
        (
            api_base,
            dynamic_api_key,
            custom_llm_provider,
        ) = dheera_ai.GithubCopilotConfig()._get_openai_compatible_provider_info(
            model, api_base, api_key, custom_llm_provider
        )
    elif custom_llm_provider == "novita":
        api_base = (
            api_base
            or get_secret("NOVITA_API_BASE")
            or "https://api.novita.ai/v3/openai"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NOVITA_API_KEY")
    elif custom_llm_provider == "snowflake":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.SnowflakeConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "gradient_ai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.GradientAIConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "featherless_ai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.FeatherlessAIConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "nscale":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.NscaleConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
    elif custom_llm_provider == "heroku":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.HerokuChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "dashscope":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.DashScopeChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "moonshot":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.MoonshotChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    # publicai is now handled by JSON config (see dheera_ai/llms/openai_like/providers.json)
    elif custom_llm_provider == "docker_model_runner":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.DockerModelRunnerChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "v0":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.V0ChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "morph":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.MorphChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "lambda_ai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.LambdaAIChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "hyperbolic":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.HyperbolicChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "vercel_ai_gateway":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.VercelAIGatewayConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "aiml":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.AIMLChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "wandb":
        api_base = (
            api_base
            or get_secret("WANDB_API_BASE")
            or "https://api.inference.wandb.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("WANDB_API_KEY")
    elif custom_llm_provider == "lemonade":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.LemonadeChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "clarifai":
        (
            api_base,
            dynamic_api_key,
        ) = dheera_ai.ClarifaiConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "ragflow":
        full_model = f"ragflow/{model}"
        (
            api_base,
            dynamic_api_key,
            _,
        ) = dheera_ai.RAGFlowConfig()._get_openai_compatible_provider_info(
            full_model, api_base, api_key, "ragflow"
        )
        model = full_model
    elif custom_llm_provider == "langgraph":
        # LangGraph is a custom provider, just need to set api_base
        api_base = (
            api_base
            or get_secret_str("LANGGRAPH_API_BASE")
            or "http://localhost:2024"
        )
        dynamic_api_key = api_key or get_secret_str("LANGGRAPH_API_KEY")

    if api_base is not None and not isinstance(api_base, str):
        raise Exception("api base needs to be a string. api_base={}".format(api_base))
    if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
        raise Exception(
            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                dynamic_api_key
            )
        )
    if dynamic_api_key is None and api_key is not None:
        dynamic_api_key = api_key
    return model, custom_llm_provider, dynamic_api_key, api_base
