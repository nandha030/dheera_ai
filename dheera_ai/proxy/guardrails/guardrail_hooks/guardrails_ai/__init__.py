from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .guardrails_ai import GuardrailsAI

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    if dheera_ai_params.guard_name is None:
        raise Exception(
            "GuardrailsAIException - Please pass the Guardrails AI guard name via 'dheera_ai_params::guard_name'"
        )

    _guardrails_ai_callback = GuardrailsAI(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
        guard_name=dheera_ai_params.guard_name,
        guardrails_ai_api_input_format=getattr(
            dheera_ai_params, "guardrails_ai_api_input_format", "llmOutput"
        ),
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_guardrails_ai_callback)

    return _guardrails_ai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GUARDRAILS_AI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.GUARDRAILS_AI.value: GuardrailsAI,
}
