from typing import TYPE_CHECKING, Union

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .prompt_shield import AzureContentSafetyPromptShieldGuardrail
from .text_moderation import AzureContentSafetyTextModerationGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai

    if not dheera_ai_params.api_key:
        raise ValueError("Azure Content Safety: api_key is required")
    if not dheera_ai_params.api_base:
        raise ValueError("Azure Content Safety: api_base is required")

    azure_guardrail = dheera_ai_params.guardrail.split("/")[1]

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Azure Content Safety: guardrail_name is required")

    if azure_guardrail == "prompt_shield":
        azure_content_safety_guardrail: Union[
            AzureContentSafetyPromptShieldGuardrail,
            AzureContentSafetyTextModerationGuardrail,
        ] = AzureContentSafetyPromptShieldGuardrail(
            guardrail_name=guardrail_name,
            **{
                **dheera_ai_params.model_dump(exclude_none=True),
                "api_key": dheera_ai_params.api_key,
                "api_base": dheera_ai_params.api_base,
                "default_on": dheera_ai_params.default_on,
                "event_hook": dheera_ai_params.mode,
            },
        )
    elif azure_guardrail == "text_moderations":
        azure_content_safety_guardrail = AzureContentSafetyTextModerationGuardrail(
            guardrail_name=guardrail_name,
            **{
                **dheera_ai_params.model_dump(exclude_none=True),
                "api_key": dheera_ai_params.api_key,
                "api_base": dheera_ai_params.api_base,
                "default_on": dheera_ai_params.default_on,
                "event_hook": dheera_ai_params.mode,
            },
        )
    else:
        raise ValueError(
            f"Azure Content Safety: {azure_guardrail} is not a valid guardrail"
        )

    dheera_ai.logging_callback_manager.add_dheera_ai_callback(
        azure_content_safety_guardrail
    )
    return azure_content_safety_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AZURE_PROMPT_SHIELD.value: initialize_guardrail,
    SupportedGuardrailIntegrations.AZURE_TEXT_MODERATIONS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.AZURE_PROMPT_SHIELD.value: AzureContentSafetyPromptShieldGuardrail,
    SupportedGuardrailIntegrations.AZURE_TEXT_MODERATIONS.value: AzureContentSafetyTextModerationGuardrail,
}
