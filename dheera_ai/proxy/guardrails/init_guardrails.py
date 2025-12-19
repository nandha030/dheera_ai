from typing import Dict, List, Optional, cast

import dheera_ai
from dheera_ai._logging import verbose_proxy_logger
from dheera_ai.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

# v2 implementation
from dheera_ai.types.guardrails import Guardrail, GuardrailItem, GuardrailItemSpec

all_guardrails: List[GuardrailItem] = []

"""
Map guardrail_name: <pre_call>, <post_call>, during_call

"""


def init_guardrails_v2(
    all_guardrails: List[Dict],
    config_file_path: Optional[str] = None,
):
    from dheera_ai.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    guardrail_list: List[Guardrail] = []

    for guardrail in all_guardrails:
        initialized_guardrail = IN_MEMORY_GUARDRAIL_HANDLER.initialize_guardrail(
            guardrail=cast(Guardrail, guardrail),
            config_file_path=config_file_path,
        )
        if initialized_guardrail:
            guardrail_list.append(initialized_guardrail)

    verbose_proxy_logger.debug(f"\nGuardrail List:{guardrail_list}\n")


### LEGACY IMPLEMENTATION ###
def initialize_guardrails(
    guardrails_config: List[Dict[str, GuardrailItemSpec]],
    premium_user: bool,
    config_file_path: str,
    dheera_ai_settings: dict,
) -> Dict[str, GuardrailItem]:
    try:
        verbose_proxy_logger.debug(f"validating  guardrails passed {guardrails_config}")
        global all_guardrails
        for item in guardrails_config:
            """
            one item looks like this:

            {'prompt_injection': {'callbacks': ['lakera_prompt_injection', 'prompt_injection_api_2'], 'default_on': True, 'enabled_roles': ['user']}}
            """
            for k, v in item.items():
                guardrail_item = GuardrailItem(**v, guardrail_name=k)
                all_guardrails.append(guardrail_item)
                dheera_ai.guardrail_name_config_map[k] = guardrail_item

        # set appropriate callbacks if they are default on
        default_on_callbacks = set()
        callback_specific_params = {}
        for guardrail in all_guardrails:
            verbose_proxy_logger.debug(guardrail.guardrail_name)
            verbose_proxy_logger.debug(guardrail.default_on)

            callback_specific_params.update(guardrail.callback_args)

            if guardrail.default_on is True:
                # add these to dheera_ai callbacks if they don't exist
                for callback in guardrail.callbacks:
                    if callback not in dheera_ai.callbacks:
                        default_on_callbacks.add(callback)

                    if guardrail.logging_only is True:
                        if callback == "presidio":
                            callback_specific_params["presidio"] = {"logging_only": True}  # type: ignore

        default_on_callbacks_list = list(default_on_callbacks)
        if len(default_on_callbacks_list) > 0:
            initialize_callbacks_on_proxy(
                value=default_on_callbacks_list,
                premium_user=premium_user,
                config_file_path=config_file_path,
                dheera_ai_settings=dheera_ai_settings,
                callback_specific_params=callback_specific_params,
            )

        return dheera_ai.guardrail_name_config_map
    except Exception as e:
        verbose_proxy_logger.exception(
            "error initializing guardrails {}".format(str(e))
        )
        raise e
