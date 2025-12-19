"""
Unit test for DheeraAI Proxy Responses API configuration.
"""

import pytest

from dheera_ai.types.utils import LlmProviders
from dheera_ai.utils import ProviderConfigManager


def test_dheera_ai_proxy_responses_api_config():
    """Test that dheera_ai_proxy provider returns correct Responses API config"""
    from dheera_ai.llms.dheera_ai_proxy.responses.transformation import (
        DheeraAIProxyResponsesAPIConfig,
    )

    config = ProviderConfigManager.get_provider_responses_api_config(
        model="dheera_ai_proxy/gpt-4",
        provider=LlmProviders.DHEERA_AI_PROXY,
    )
    print(f"config: {config}")
    assert config is not None, "Config should not be None for dheera_ai_proxy provider"
    assert isinstance(
        config, DheeraAIProxyResponsesAPIConfig
    ), f"Expected DheeraAIProxyResponsesAPIConfig, got {type(config)}"
    assert (
        config.custom_llm_provider == LlmProviders.DHEERA_AI_PROXY
    ), "custom_llm_provider should be DHEERA_AI_PROXY"


def test_dheera_ai_proxy_responses_api_config_get_complete_url():
    """Test that get_complete_url works correctly"""
    import os
    from dheera_ai.llms.dheera_ai_proxy.responses.transformation import (
        DheeraAIProxyResponsesAPIConfig,
    )

    config = DheeraAIProxyResponsesAPIConfig()

    # Test with explicit api_base
    url = config.get_complete_url(
        api_base="https://my-proxy.example.com",
        dheera_ai_params={},
    )
    assert url == "https://my-proxy.example.com/responses"

    # Test with trailing slash
    url = config.get_complete_url(
        api_base="https://my-proxy.example.com/",
        dheera_ai_params={},
    )
    assert url == "https://my-proxy.example.com/responses"

    # Test that it raises error when api_base is None and env var is not set
    if "DHEERA_AI_PROXY_API_BASE" in os.environ:
        del os.environ["DHEERA_AI_PROXY_API_BASE"]
    
    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(api_base=None, dheera_ai_params={})


def test_dheera_ai_proxy_responses_api_config_inherits_from_openai():
    """Test that DheeraAIProxyResponsesAPIConfig extends OpenAI config properly"""
    from dheera_ai.llms.dheera_ai_proxy.responses.transformation import (
        DheeraAIProxyResponsesAPIConfig,
    )
    from dheera_ai.llms.openai.responses.transformation import (
        OpenAIResponsesAPIConfig,
    )

    config = DheeraAIProxyResponsesAPIConfig()
    
    # Should inherit from OpenAI config
    assert isinstance(config, OpenAIResponsesAPIConfig)
    
    # Should have the correct provider set
    assert config.custom_llm_provider == LlmProviders.DHEERA_AI_PROXY


if __name__ == "__main__":
    test_dheera_ai_proxy_responses_api_config()
    test_dheera_ai_proxy_responses_api_config_get_complete_url()
    test_dheera_ai_proxy_responses_api_config_inherits_from_openai()
    print("All tests passed!")
