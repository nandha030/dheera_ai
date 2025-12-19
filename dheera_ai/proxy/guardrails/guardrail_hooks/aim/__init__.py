from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .aim import AimGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai
    from dheera_ai.proxy.guardrails.guardrail_hooks.aim import AimGuardrail

    _aim_callback = AimGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_aim_callback)

    return _aim_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AIM.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.AIM.value: AimGuardrail,
}
