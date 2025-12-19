# dheera_ai/proxy/guardrails/guardrail_initializers.py
from typing import Any, Dict, List, Optional

import dheera_ai
from dheera_ai.proxy._types import CommonProxyErrors
from dheera_ai.types.guardrails import *


def initialize_bedrock(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

    _bedrock_callback = BedrockGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        guardrailIdentifier=dheera_ai_params.guardrailIdentifier,
        guardrailVersion=dheera_ai_params.guardrailVersion,
        default_on=dheera_ai_params.default_on,
        disable_exception_on_block=dheera_ai_params.disable_exception_on_block,
        mask_request_content=dheera_ai_params.mask_request_content,
        mask_response_content=dheera_ai_params.mask_response_content,
        aws_region_name=dheera_ai_params.aws_region_name,
        aws_access_key_id=dheera_ai_params.aws_access_key_id,
        aws_secret_access_key=dheera_ai_params.aws_secret_access_key,
        aws_session_token=dheera_ai_params.aws_session_token,
        aws_session_name=dheera_ai_params.aws_session_name,
        aws_profile_name=dheera_ai_params.aws_profile_name,
        aws_role_name=dheera_ai_params.aws_role_name,
        aws_web_identity_token=dheera_ai_params.aws_web_identity_token,
        aws_sts_endpoint=dheera_ai_params.aws_sts_endpoint,
        aws_bedrock_runtime_endpoint=dheera_ai_params.aws_bedrock_runtime_endpoint,
        experimental_use_latest_role_message_only=dheera_ai_params.experimental_use_latest_role_message_only,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_bedrock_callback)
    return _bedrock_callback


def initialize_lakera(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation

    _lakera_callback = lakeraAI_Moderation(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        category_thresholds=dheera_ai_params.category_thresholds,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_lakera_callback)
    return _lakera_callback


def initialize_lakera_v2(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail

    _lakera_v2_callback = LakeraAIGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
        project_id=dheera_ai_params.project_id,
        payload=dheera_ai_params.payload,
        breakdown=dheera_ai_params.breakdown,
        metadata=dheera_ai_params.metadata,
        dev_info=dheera_ai_params.dev_info,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_lakera_v2_callback)
    return _lakera_v2_callback


def initialize_presidio(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.presidio import (
        _OPTIONAL_PresidioPIIMasking,
    )

    filter_scope = getattr(dheera_ai_params, "presidio_filter_scope", None) or "both"
    run_input = filter_scope in ("input", "both")
    run_output = filter_scope in ("output", "both")

    def _make_presidio_callback(**overrides):
        params = dict(
            guardrail_name=guardrail.get("guardrail_name", ""),
            event_hook=dheera_ai_params.mode,
            output_parse_pii=dheera_ai_params.output_parse_pii,
            presidio_ad_hoc_recognizers=dheera_ai_params.presidio_ad_hoc_recognizers,
            mock_redacted_text=dheera_ai_params.mock_redacted_text,
            default_on=dheera_ai_params.default_on,
            pii_entities_config=dheera_ai_params.pii_entities_config,
            presidio_score_thresholds=dheera_ai_params.presidio_score_thresholds,
            presidio_analyzer_api_base=dheera_ai_params.presidio_analyzer_api_base,
            presidio_anonymizer_api_base=dheera_ai_params.presidio_anonymizer_api_base,
            presidio_language=dheera_ai_params.presidio_language,
            apply_to_output=False,
        )
        params.update(overrides)
        callback = _OPTIONAL_PresidioPIIMasking(**params)
        dheera_ai.logging_callback_manager.add_dheera_ai_callback(callback)
        return callback

    primary_callback = None

    if run_input:
        primary_callback = _make_presidio_callback()

        if dheera_ai_params.output_parse_pii:
            _make_presidio_callback(
                output_parse_pii=True,
                event_hook=GuardrailEventHooks.post_call.value,
            )

    if run_output:
        output_callback = _make_presidio_callback(
            apply_to_output=True,
            event_hook=GuardrailEventHooks.post_call.value,
            output_parse_pii=False,
        )
        if primary_callback is None:
            primary_callback = output_callback

    return primary_callback


def initialize_hide_secrets(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    try:
        from dheera_ai_enterprise.enterprise_callbacks.secret_detection import (
            _ENTERPRISE_SecretDetection,
        )
    except ImportError:
        raise Exception(
            "Trying to use Secret Detection"
            + CommonProxyErrors.missing_enterprise_package.value
        )

    _secret_detection_object = _ENTERPRISE_SecretDetection(
        detect_secrets_config=dheera_ai_params.detect_secrets_config,
        event_hook=dheera_ai_params.mode,
        guardrail_name=guardrail.get("guardrail_name", ""),
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_secret_detection_object)
    return _secret_detection_object


def initialize_tool_permission(dheera_ai_params: LitellmParams, guardrail: Guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.tool_permission import (
        ToolPermissionGuardrail,
    )

    rules: Optional[List[Dict[str, Any]]] = None
    if dheera_ai_params.rules:
        rules = []
        for rule in dheera_ai_params.rules:
            if hasattr(rule, "model_dump"):
                rules.append(rule.model_dump())
            else:
                rules.append(dict(rule))

    _tool_permission_callback = ToolPermissionGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        rules=rules,
        default_action=getattr(dheera_ai_params, "default_action", "deny"),
        on_disallowed_action=getattr(dheera_ai_params, "on_disallowed_action", "block"),
        default_on=dheera_ai_params.default_on,
        violation_message_template=dheera_ai_params.violation_message_template,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_tool_permission_callback)
    return _tool_permission_callback


def initialize_lasso(
    dheera_ai_params: LitellmParams,
    guardrail: Guardrail,
):
    from dheera_ai.proxy.guardrails.guardrail_hooks.lasso import LassoGuardrail

    _lasso_callback = LassoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        lasso_api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base,
        user_id=dheera_ai_params.lasso_user_id,
        conversation_id=dheera_ai_params.lasso_conversation_id,
        mask=dheera_ai_params.mask,
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_lasso_callback)

    return _lasso_callback


def initialize_panw_prisma_airs(dheera_ai_params, guardrail):
    from dheera_ai.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
        PanwPrismaAirsHandler,
    )

    if not dheera_ai_params.api_key:
        raise ValueError("PANW Prisma AIRS: api_key is required")
    if not dheera_ai_params.profile_name:
        raise ValueError("PANW Prisma AIRS: profile_name is required")

    _panw_callback = PanwPrismaAirsHandler(
        guardrail_name=guardrail.get(
            "guardrail_name", "panw_prisma_airs"
        ),  # Use .get() with default
        api_key=dheera_ai_params.api_key,
        api_base=dheera_ai_params.api_base
        or "https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request",
        profile_name=dheera_ai_params.profile_name,
        default_on=dheera_ai_params.default_on,
        mask_on_block=getattr(dheera_ai_params, "mask_on_block", False),
        mask_request_content=getattr(dheera_ai_params, "mask_request_content", False),
        mask_response_content=getattr(dheera_ai_params, "mask_response_content", False),
        app_name=getattr(dheera_ai_params, "app_name", None),
        fallback_on_error=getattr(dheera_ai_params, "fallback_on_error", "block"),
        timeout=float(getattr(dheera_ai_params, "timeout", 10.0)),
    )
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_panw_callback)

    return _panw_callback
