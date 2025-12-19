from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .javelin import JavelinGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    if dheera_ai_params.guard_name is None:
        raise Exception(
            "JavelinGuardrailException - Please pass the Javelin guard name via 'dheera_ai_params::guard_name'"
        )

    _javelin_callback = JavelinGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        javelin_guard_name=dheera_ai_params.guard_name,
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on or False,
        api_version=dheera_ai_params.api_version or "v1",
        config=dheera_ai_params.config,
        metadata=dheera_ai_params.metadata,
        application=dheera_ai_params.application,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_javelin_callback)

    return _javelin_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.JAVELIN.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.JAVELIN.value: JavelinGuardrail,
}
