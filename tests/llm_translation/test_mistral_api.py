import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import dheera_ai.types
import dheera_ai.types.utils
from dheera_ai.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import dheera_ai

from dheera_ai.llms.anthropic.common_utils import process_anthropic_headers
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest


@pytest.mark.flaky(retries=3, delay=2)
class TestMistralCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        dheera_ai.set_verbose = True
        return {"model": "mistral/mistral-medium-latest"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/dheera_ai/issues/6833"""
        pass
