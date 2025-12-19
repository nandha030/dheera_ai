from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .lasso import LassoGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _lasso_callback = LassoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base,
        user_id=dheera_ai_params.lasso_user_id,
        conversation_id=dheera_ai_params.lasso_conversation_id,
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_lasso_callback)

    return _lasso_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.LASSO.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.LASSO.value: LassoGuardrail,
}
