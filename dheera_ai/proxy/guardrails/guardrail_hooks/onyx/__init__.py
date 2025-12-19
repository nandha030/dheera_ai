from typing import TYPE_CHECKING

from dheera_ai.proxy.guardrails.guardrail_hooks.onyx.onyx import OnyxGuardrail
from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _onyx_callback = OnyxGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_onyx_callback)

    return _onyx_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ONYX.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ONYX.value: OnyxGuardrail,
}
