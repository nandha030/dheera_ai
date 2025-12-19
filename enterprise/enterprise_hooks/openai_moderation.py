# +-------------------------------------------------------------+
#
#           Use OpenAI /moderations for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import sys

from fastapi import HTTPException

import dheera_ai
from dheera_ai._logging import verbose_proxy_logger
from dheera_ai.integrations.custom_logger import CustomLogger
from dheera_ai.proxy._types import UserAPIKeyAuth
from dheera_ai.types.utils import CallTypesLiteral


class _ENTERPRISE_OpenAI_Moderation(CustomLogger):
    def __init__(self):
        self.model_name = (
            dheera_ai.openai_moderations_model_name or "text-moderation-latest"
        )  # pass the model_name you initialized on dheera_ai.Router()
        pass

    #### CALL HOOKS - proxy only ####

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        text = ""
        if "messages" in data and isinstance(data["messages"], list):
            for m in data["messages"]:  # assume messages is a list
                if "content" in m and isinstance(m["content"], str):
                    text += m["content"]

        from dheera_ai.proxy.proxy_server import llm_router

        if llm_router is None:
            return

        moderation_response = await llm_router.amoderation(
            model=self.model_name, input=text
        )

        verbose_proxy_logger.debug("Moderation response: %s", moderation_response)
        if moderation_response and moderation_response.results[0].flagged is True:
            raise HTTPException(
                status_code=403, detail={"error": "Violated content safety policy"}
            )
        pass
