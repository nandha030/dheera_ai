from typing import TYPE_CHECKING, Any, Optional, Union

from dheera_ai._logging import verbose_router_logger
from dheera_ai.constants import MAX_EXCEPTION_MESSAGE_LENGTH
from dheera_ai.router_utils.cooldown_handlers import (
    _async_get_cooldown_deployments_with_debug_info,
)
from dheera_ai.types.integrations.slack_alerting import AlertType
from dheera_ai.types.router import RouterRateLimitError

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from dheera_ai.router import Router as _Router

    LitellmRouter = _Router
    Span = Union[_Span, Any]
else:
    LitellmRouter = Any
    Span = Any


async def send_llm_exception_alert(
    dheera_ai_router_instance: LitellmRouter,
    request_kwargs: dict,
    error_traceback_str: str,
    original_exception,
):
    """
    Only runs if router.slack_alerting_logger is set
    Sends a Slack / MS Teams alert for the LLM API call failure. Only if router.slack_alerting_logger is set.

    Parameters:
        dheera_ai_router_instance (_Router): The LitellmRouter instance.
        original_exception (Any): The original exception that occurred.

    Returns:
        None
    """
    if dheera_ai_router_instance is None:
        return

    if not hasattr(dheera_ai_router_instance, "slack_alerting_logger"):
        return

    if dheera_ai_router_instance.slack_alerting_logger is None:
        return

    if "proxy_server_request" in request_kwargs:
        # Do not send any alert if it's a request from dheera_ai proxy server request
        # the proxy is already instrumented to send LLM API call failures
        return

    dheera_ai_debug_info = getattr(original_exception, "dheera_ai_debug_info", None)
    exception_str = str(original_exception)
    if dheera_ai_debug_info is not None:
        exception_str += dheera_ai_debug_info
    exception_str += f"\n\n{error_traceback_str[:MAX_EXCEPTION_MESSAGE_LENGTH]}"

    await dheera_ai_router_instance.slack_alerting_logger.send_alert(
        message=f"LLM API call failed: `{exception_str}`",
        level="High",
        alert_type=AlertType.llm_exceptions,
        alerting_metadata={},
    )


async def async_raise_no_deployment_exception(
    dheera_ai_router_instance: LitellmRouter, model: str, parent_otel_span: Optional[Span]
):
    """
    Raises a RouterRateLimitError if no deployment is found for the given model.
    """
    verbose_router_logger.info(
        f"get_available_deployment for model: {model}, No deployment available"
    )
    model_ids = dheera_ai_router_instance.get_model_ids(model_name=model)
    _cooldown_time = dheera_ai_router_instance.cooldown_cache.get_min_cooldown(
        model_ids=model_ids, parent_otel_span=parent_otel_span
    )
    _cooldown_list = await _async_get_cooldown_deployments_with_debug_info(
        dheera_ai_router_instance=dheera_ai_router_instance,
        parent_otel_span=parent_otel_span,
    )
    verbose_router_logger.info(
        f"No deployment found for model: {model}, cooldown_list with debug info: {_cooldown_list}"
    )

    cooldown_list_ids = [cooldown_model[0] for cooldown_model in (_cooldown_list or [])]
    return RouterRateLimitError(
        model=model,
        cooldown_time=_cooldown_time,
        enable_pre_call_checks=dheera_ai_router_instance.enable_pre_call_checks,
        cooldown_list=cooldown_list_ids,
    )
