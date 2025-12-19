"""OpenAI Chat Completions message handler for Unified Guardrails."""

from dheera_ai.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from dheera_ai.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.completion: OpenAIChatCompletionsHandler,
    CallTypes.acompletion: OpenAIChatCompletionsHandler,
}
__all__ = ["guardrail_translation_mappings"]
