from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .generic_guardrail_api import GenericGuardrailAPI

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _generic_guardrail_api_callback = GenericGuardrailAPI(
        api_base=dheera_ai_params.api_base,
        headers=getattr(dheera_ai_params, "headers", None),
        additional_provider_specific_params=getattr(
            dheera_ai_params, "additional_provider_specific_params", {}
        ),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )

    dheera_ai.logging_callback_manager.add_dheera_ai_callback(
        _generic_guardrail_api_callback
    )
    return _generic_guardrail_api_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: GenericGuardrailAPI,
}
