from typing import TYPE_CHECKING

import dheera_ai
from dheera_ai.proxy.guardrails.guardrail_hooks.openai.moderations import (
    OpenAIModerationGuardrail,
)
from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("OpenAI Moderation: guardrail_name is required")
    
    openai_moderation_guardrail = OpenAIModerationGuardrail(
        guardrail_name=guardrail_name,
        **{
            **dheera_ai_params.model_dump(exclude_none=True),
            "api_key": dheera_ai_params.api_key,
            "api_base": dheera_ai_params.api_base,
            "default_on": dheera_ai_params.default_on,
            "event_hook": dheera_ai_params.mode,
            "model": dheera_ai_params.model,
        },
    )

    dheera_ai.logging_callback_manager.add_dheera_ai_callback(
        openai_moderation_guardrail
    )

    return openai_moderation_guardrail



guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: OpenAIModerationGuardrail,
}
