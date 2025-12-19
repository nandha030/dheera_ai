from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .pangea import PangeaHandler

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Pangea guardrail name is required")

    _pangea_callback = PangeaHandler(
        guardrail_name=guardrail_name,
        pangea_input_recipe=dheera_ai_params.pangea_input_recipe,
        pangea_output_recipe=dheera_ai_params.pangea_output_recipe,
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_pangea_callback)

    return _pangea_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PANGEA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PANGEA.value: PangeaHandler,
}
