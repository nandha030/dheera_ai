from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .model_armor import ModelArmorGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai
    from dheera_ai.proxy.guardrails.guardrail_hooks.model_armor import (
        ModelArmorGuardrail,
    )

    _model_armor_callback = ModelArmorGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        template_id=dheera_ai_params.template_id,
        project_id=dheera_ai_params.project_id,
        location=dheera_ai_params.location,
        credentials=dheera_ai_params.credentials,
        api_endpoint=dheera_ai_params.api_endpoint,
        default_on=dheera_ai_params.default_on,
        mask_request_content=dheera_ai_params.mask_request_content,
        mask_response_content=dheera_ai_params.mask_response_content,
        fail_on_error=dheera_ai_params.fail_on_error,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_model_armor_callback)

    return _model_armor_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MODEL_ARMOR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.MODEL_ARMOR.value: ModelArmorGuardrail,
}
