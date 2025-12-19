"""
Handler for transforming responses api requests to dheera_ai.completion requests
"""

from typing import Any, Coroutine, Dict, Optional, Union

import dheera_ai
from dheera_ai.responses.dheera_ai_completion_transformation.streaming_iterator import (
    DheeraAICompletionStreamingIterator,
)
from dheera_ai.responses.dheera_ai_completion_transformation.transformation import (
    DheeraAICompletionResponsesConfig,
)
from dheera_ai.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from dheera_ai.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from dheera_ai.types.utils import ModelResponse


class DheeraAICompletionTransformationHandler:

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        dheera_ai_completion_request: dict = (
            DheeraAICompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                dheera_ai_completion_request=dheera_ai_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(dheera_ai_completion_request)

        dheera_ai_completion_response: Union[
            ModelResponse, dheera_ai.CustomStreamWrapper
        ] = dheera_ai.completion(
            **dheera_ai_completion_request,
            **kwargs,
        )

        if isinstance(dheera_ai_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                DheeraAICompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=dheera_ai_completion_response,
                    request_input=input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(dheera_ai_completion_response, dheera_ai.CustomStreamWrapper):
            return DheeraAICompletionStreamingIterator(
                model=model,
                dheera_ai_custom_stream_wrapper=dheera_ai_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                dheera_ai_metadata=kwargs.get("dheera_ai_metadata", {}),
            )

    async def async_response_api_handler(
        self,
        dheera_ai_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:

        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            dheera_ai_completion_request = await DheeraAICompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                dheera_ai_completion_request=dheera_ai_completion_request,
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(dheera_ai_completion_request)

        dheera_ai_completion_response: Union[
            ModelResponse, dheera_ai.CustomStreamWrapper
        ] = await dheera_ai.acompletion(
            **acompletion_args,
        )

        if isinstance(dheera_ai_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                DheeraAICompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=dheera_ai_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(dheera_ai_completion_response, dheera_ai.CustomStreamWrapper):
            return DheeraAICompletionStreamingIterator(
                model=dheera_ai_completion_request.get("model") or "",
                dheera_ai_custom_stream_wrapper=dheera_ai_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=dheera_ai_completion_request.get(
                    "custom_llm_provider"
                ),
                dheera_ai_metadata=kwargs.get("dheera_ai_metadata", {}),
            )
