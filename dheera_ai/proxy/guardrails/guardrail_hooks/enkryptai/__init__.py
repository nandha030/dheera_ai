from .enkryptai import EnkryptAIGuardrails

__all__ = ["EnkryptAIGuardrails"]


from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    _enkryptai_callback = EnkryptAIGuardrails(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base,
        policy_name=dheera_ai_params.policy_name,
        deployment_name=dheera_ai_params.deployment_name,
        detectors=dheera_ai_params.detectors,
        block_on_violation=dheera_ai_params.block_on_violation,
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_enkryptai_callback)

    return _enkryptai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ENKRYPTAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ENKRYPTAI.value: EnkryptAIGuardrails,
}

