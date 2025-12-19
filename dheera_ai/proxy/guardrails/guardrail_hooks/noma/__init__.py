from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .noma import NomaGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _noma_callback = NomaGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base,
        application_id=dheera_ai_params.application_id,
        monitor_mode=dheera_ai_params.monitor_mode,
        block_failures=dheera_ai_params.block_failures,
        anonymize_input=dheera_ai_params.anonymize_input,
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_noma_callback)

    return _noma_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.NOMA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.NOMA.value: NomaGuardrail,
}
