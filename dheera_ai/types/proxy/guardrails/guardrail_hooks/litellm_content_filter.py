from dheera_ai.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class LitellmContentFilterGuardrailConfigModel(GuardrailConfigModel):
    @staticmethod
    def ui_friendly_name() -> str:
        return "DheeraAI Content Filter"