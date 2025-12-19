from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .hiddenlayer import HiddenlayerGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    api_id = dheera_ai_params.api_id if hasattr(dheera_ai_params, "api_id") else None
    auth_url = dheera_ai_params.auth_url if hasattr(dheera_ai_params, "auth_url") else None

    _hiddenlayer_callback = HiddenlayerGuardrail(
        api_base=dheera_ai_params.api_base,
        api_id=api_id,
        api_key=dheera_ai_params.api_key,
        auth_url=auth_url,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )

    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_hiddenlayer_callback)
    return _hiddenlayer_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: HiddenlayerGuardrail,
}
