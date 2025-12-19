from typing import TYPE_CHECKING

import dheera_ai
from dheera_ai.proxy.guardrails.guardrail_hooks.dheera_ai_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    """
    Initialize the Content Filter Guardrail.
    
    Args:
        dheera_ai_params: Guardrail configuration parameters
        guardrail: Guardrail metadata
        
    Returns:
        Initialized ContentFilterGuardrail instance
    """
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Content Filter: guardrail_name is required")
    
    content_filter_guardrail = ContentFilterGuardrail(
        guardrail_name=guardrail_name,
        patterns=dheera_ai_params.patterns,
        blocked_words=dheera_ai_params.blocked_words,
        blocked_words_file=dheera_ai_params.blocked_words_file,
        event_hook=dheera_ai_params.mode,  # type: ignore
        default_on=dheera_ai_params.default_on or False,
    )
    
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(
        content_filter_guardrail
    )
    
    return content_filter_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.DHEERA_AI_CONTENT_FILTER.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.DHEERA_AI_CONTENT_FILTER.value: ContentFilterGuardrail,
}

