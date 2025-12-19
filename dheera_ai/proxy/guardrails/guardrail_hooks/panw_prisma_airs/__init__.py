from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .panw_prisma_airs import PanwPrismaAirsHandler

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    guardrail_name = guardrail.get("guardrail_name")

    # Note: api_key and profile_name can be None - handler will use env vars or API key's linked profile
    if not guardrail_name:
        raise ValueError("PANW Prisma AIRS: guardrail_name is required")

    _panw_callback = PanwPrismaAirsHandler(
        **{
            **dheera_ai_params.model_dump(),
            "guardrail_name": guardrail_name,
            "event_hook": dheera_ai_params.mode,
            "default_on": dheera_ai_params.default_on or False,
        }
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_panw_callback)

    return _panw_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: PanwPrismaAirsHandler,
}
