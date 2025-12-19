"""OpenAI Text-to-Speech handler for Unified Guardrails."""

from dheera_ai.llms.openai.speech.guardrail_translation.handler import (
    OpenAITextToSpeechHandler,
)
from dheera_ai.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.speech: OpenAITextToSpeechHandler,
    CallTypes.aspeech: OpenAITextToSpeechHandler,
}

__all__ = ["guardrail_translation_mappings", "OpenAITextToSpeechHandler"]
