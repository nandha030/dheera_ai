from typing import Optional
from unittest.mock import patch

import pytest

import dheera_ai
from dheera_ai.llms.dheera_ai_proxy.chat.transformation import DheeraAIProxyChatConfig


def test_dheera_ai_proxy_chat_transformation():
    """
    Assert messages are not transformed when calling dheera_ai proxy
    """
    config = DheeraAIProxyChatConfig()
    file_content = [
        {"type": "text", "text": "What is this document about?"},
        {
            "type": "file",
            "file": {
                "file_id": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "format": "application/pdf",
            },
        },
    ]
    messages = [{"role": "user", "content": file_content}]
    assert config.transform_request(
        model="model",
        messages=messages,
        optional_params={},
        dheera_ai_params={},
        headers={},
    ) == {"model": "model", "messages": messages}


def test_dheera_ai_gateway_from_sdk_with_user_param():
    from dheera_ai.llms.dheera_ai_proxy.chat.transformation import DheeraAIProxyChatConfig

    supported_params = DheeraAIProxyChatConfig().get_supported_openai_params(
        "openai/gpt-4o"
    )
    print(f"supported_params: {supported_params}")
    assert "user" in supported_params
