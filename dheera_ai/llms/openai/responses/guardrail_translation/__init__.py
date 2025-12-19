"""OpenAI Responses API handler for Unified Guardrails."""

from dheera_ai.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
from dheera_ai.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.responses: OpenAIResponsesHandler,
    CallTypes.aresponses: OpenAIResponsesHandler,
}
__all__ = ["guardrail_translation_mappings"]
