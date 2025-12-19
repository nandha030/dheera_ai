"""Gray Swan Cygnal guardrail integration for DheeraAI."""

from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .grayswan import (
    GraySwanGuardrail,
    GraySwanGuardrailAPIError,
    GraySwanGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    dheera_ai_params: "LitellmParams", guardrail: "Guardrail"
) -> GraySwanGuardrail:
    import dheera_ai

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Gray Swan guardrail requires a guardrail_name")

    optional_params = getattr(dheera_ai_params, "optional_params", None)

    grayswan_guardrail = GraySwanGuardrail(
        guardrail_name=guardrail_name,
        api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base,
        on_flagged_action=_get_config_value(
            dheera_ai_params, optional_params, "on_flagged_action"
        ),
        violation_threshold=_get_config_value(
            dheera_ai_params, optional_params, "violation_threshold"
        ),
        reasoning_mode=_get_config_value(
            dheera_ai_params, optional_params, "reasoning_mode"
        ),
        categories=_get_config_value(dheera_ai_params, optional_params, "categories"),
        policy_id=_get_config_value(dheera_ai_params, optional_params, "policy_id"),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )

    dheera_ai.logging_callback_manager.add_dheera_ai_callback(grayswan_guardrail)
    return grayswan_guardrail


def _get_config_value(dheera_ai_params, optional_params, attribute_name):
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(dheera_ai_params, attribute_name, None)


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GRAYSWAN.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.GRAYSWAN.value: GraySwanGuardrail,
}


__all__ = [
    "GraySwanGuardrail",
    "GraySwanGuardrailAPIError",
    "GraySwanGuardrailMissingSecrets",
    "initialize_guardrail",
]
