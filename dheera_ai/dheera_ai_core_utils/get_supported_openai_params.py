from typing import Literal, Optional

import dheera_ai
from dheera_ai.exceptions import BadRequestError
from dheera_ai.types.utils import LlmProviders, LlmProvidersSet


def get_supported_openai_params(  # noqa: PLR0915
    model: str,
    custom_llm_provider: Optional[str] = None,
    request_type: Literal[
        "chat_completion", "embeddings", "transcription"
    ] = "chat_completion",
) -> Optional[list]:
    """
    Returns the supported openai params for a given model + provider

    Example:
    ```
    get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")
    ```

    Returns:
    - List if custom_llm_provider is mapped
    - None if unmapped
    """
    if not custom_llm_provider:
        try:
            custom_llm_provider = dheera_ai.get_llm_provider(model=model)[1]
        except BadRequestError:
            return None

    if custom_llm_provider in LlmProvidersSet:
        provider_config = dheera_ai.ProviderConfigManager.get_provider_chat_config(
            model=model, provider=LlmProviders(custom_llm_provider)
        )
    elif custom_llm_provider.split("/")[0] in LlmProvidersSet:
        provider_config = dheera_ai.ProviderConfigManager.get_provider_chat_config(
            model=model, provider=LlmProviders(custom_llm_provider.split("/")[0])
        )
    else:
        provider_config = None

    if provider_config and request_type == "chat_completion":
        return provider_config.get_supported_openai_params(model=model)

    if custom_llm_provider == "bedrock":
        return dheera_ai.AmazonConverseConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "meta_llama":
        provider_config = dheera_ai.ProviderConfigManager.get_provider_chat_config(
            model=model, provider=LlmProviders.LLAMA
        )
        if provider_config:
            return provider_config.get_supported_openai_params(model=model)
    elif custom_llm_provider == "ollama":
        return dheera_ai.OllamaConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ollama_chat":
        return dheera_ai.OllamaChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "anthropic":
        return dheera_ai.AnthropicConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "anthropic_text":
        return dheera_ai.AnthropicTextConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "fireworks_ai":
        if request_type == "embeddings":
            return dheera_ai.FireworksAIEmbeddingConfig().get_supported_openai_params(
                model=model
            )
        elif request_type == "transcription":
            return dheera_ai.FireworksAIAudioTranscriptionConfig().get_supported_openai_params(
                model=model
            )
        else:
            return dheera_ai.FireworksAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nvidia_nim":
        if request_type == "chat_completion":
            return dheera_ai.nvidiaNimConfig.get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return dheera_ai.nvidiaNimEmbeddingConfig.get_supported_openai_params()
    elif custom_llm_provider == "cerebras":
        return dheera_ai.CerebrasConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "baseten":
        return dheera_ai.BasetenConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "xai":
        return dheera_ai.XAIChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ai21_chat" or custom_llm_provider == "ai21":
        return dheera_ai.AI21ChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "volcengine":
        return dheera_ai.VolcEngineConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "groq":
        return dheera_ai.GroqChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "hosted_vllm":
        return dheera_ai.HostedVLLMChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "vllm":
        return dheera_ai.VLLMConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepseek":
        return dheera_ai.DeepSeekChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "cohere_chat" or custom_llm_provider == "cohere":
        return dheera_ai.CohereChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "maritalk":
        return dheera_ai.MaritalkConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "openai":
        if request_type == "transcription":
            transcription_provider_config = (
                dheera_ai.ProviderConfigManager.get_provider_audio_transcription_config(
                    model=model, provider=LlmProviders.OPENAI
                )
            )
            if isinstance(
                transcription_provider_config, dheera_ai.OpenAIGPTAudioTranscriptionConfig
            ):
                return transcription_provider_config.get_supported_openai_params(
                    model=model
                )
            else:
                raise ValueError(
                    f"Unsupported provider config: {transcription_provider_config} for model: {model}"
                )
        return dheera_ai.OpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "sap":
        if request_type == "chat_completion":
            return dheera_ai.GenAIHubOrchestrationConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return dheera_ai.GenAIHubEmbeddingConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "azure":
        if dheera_ai.AzureOpenAIO1Config().is_o_series_model(model=model):
            return dheera_ai.AzureOpenAIO1Config().get_supported_openai_params(
                model=model
            )
        elif dheera_ai.AzureOpenAIGPT5Config.is_model_gpt_5_model(model=model):
            return dheera_ai.AzureOpenAIGPT5Config().get_supported_openai_params(
                model=model
            )
        else:
            return dheera_ai.AzureOpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "openrouter":
        return dheera_ai.OpenrouterConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "vercel_ai_gateway":
        return dheera_ai.VercelAIGatewayConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "mistral" or custom_llm_provider == "codestral":
        # mistal and codestral api have the exact same params
        if request_type == "chat_completion":
            return dheera_ai.MistralConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return dheera_ai.MistralEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "text-completion-codestral":
        return dheera_ai.CodestralTextCompletionConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "sambanova":
        if request_type == "embeddings":
            dheera_ai.SambaNovaEmbeddingConfig().get_supported_openai_params(model=model)
        else:
            return dheera_ai.SambanovaConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nebius":
        if request_type == "chat_completion":
            return dheera_ai.NebiusConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "wandb":
        if request_type == "chat_completion":
            return dheera_ai.WandbConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "replicate":
        return dheera_ai.ReplicateConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "huggingface":
        return dheera_ai.HuggingFaceChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "jina_ai":
        if request_type == "embeddings":
            return dheera_ai.JinaAIEmbeddingConfig().get_supported_openai_params(
                model=model
            )
    elif custom_llm_provider == "together_ai":
        return dheera_ai.TogetherAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "databricks":
        if request_type == "chat_completion":
            return dheera_ai.DatabricksConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return dheera_ai.DatabricksEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
        return dheera_ai.GoogleAIStudioGeminiConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "novita":
        return dheera_ai.NovitaConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        if request_type == "chat_completion":
            if model.startswith("mistral"):
                return dheera_ai.MistralConfig().get_supported_openai_params(model=model)
            elif model.startswith("codestral"):
                return (
                    dheera_ai.CodestralTextCompletionConfig().get_supported_openai_params(
                        model=model
                    )
                )
            elif model.startswith("claude"):
                return dheera_ai.VertexAIAnthropicConfig().get_supported_openai_params(
                    model=model
                )
            elif model.startswith("gemini"):
                return dheera_ai.VertexGeminiConfig().get_supported_openai_params(
                    model=model
                )
            else:
                return dheera_ai.VertexAILlama3Config().get_supported_openai_params(
                    model=model
                )
        elif request_type == "embeddings":
            return dheera_ai.VertexAITextEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "sagemaker":
        return dheera_ai.SagemakerConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "aleph_alpha":
        return [
            "max_tokens",
            "stream",
            "top_p",
            "temperature",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
    elif custom_llm_provider == "cloudflare":
        return dheera_ai.CloudflareChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nlp_cloud":
        return dheera_ai.NLPCloudConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "petals":
        return dheera_ai.PetalsConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepinfra":
        return dheera_ai.DeepInfraConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "perplexity":
        return dheera_ai.PerplexityChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nscale":
        return dheera_ai.NscaleConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "anyscale":
        return [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    elif custom_llm_provider == "watsonx":
        return dheera_ai.IBMWatsonXChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "watsonx_text":
        return dheera_ai.IBMWatsonXAIConfig().get_supported_openai_params(model=model)
    elif (
        custom_llm_provider == "custom_openai"
        or custom_llm_provider == "text-completion-openai"
    ):
        return dheera_ai.OpenAITextCompletionConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "predibase":
        return dheera_ai.PredibaseConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "voyage":
        return dheera_ai.VoyageEmbeddingConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "infinity":
        return dheera_ai.InfinityEmbeddingConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "triton":
        if request_type == "embeddings":
            return dheera_ai.TritonEmbeddingConfig().get_supported_openai_params(
                model=model
            )
        else:
            return dheera_ai.TritonConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepgram":
        if request_type == "transcription":
            return (
                dheera_ai.DeepgramAudioTranscriptionConfig().get_supported_openai_params(
                    model=model
                )
            )
    elif custom_llm_provider == "ovhcloud":
        if request_type == "transcription":
            from dheera_ai.llms.ovhcloud.audio_transcription.transformation import (
                OVHCloudAudioTranscriptionConfig,
            )

            return OVHCloudAudioTranscriptionConfig().get_supported_openai_params(
                model=model
            )
    elif custom_llm_provider == "elevenlabs":
        if request_type == "transcription":
            from dheera_ai.llms.elevenlabs.audio_transcription.transformation import (
                ElevenLabsAudioTranscriptionConfig,
            )

            return ElevenLabsAudioTranscriptionConfig().get_supported_openai_params(
                model=model
            )
    elif custom_llm_provider in dheera_ai._custom_providers:
        if request_type == "chat_completion":
            provider_config = dheera_ai.ProviderConfigManager.get_provider_chat_config(
                model=model, provider=LlmProviders.CUSTOM
            )
            if provider_config:
                return provider_config.get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return None
        elif request_type == "transcription":
            return None

    return None
