from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .aporia_ai import AporiaGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _aporia_callback = AporiaGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_aporia_callback)

    return _aporia_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.APORIA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.APORIA.value: AporiaGuardrail,
}
