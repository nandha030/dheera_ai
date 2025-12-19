import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union, cast

import httpx

import dheera_ai
from dheera_ai._logging import verbose_proxy_logger
from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
from dheera_ai.llms.anthropic import get_anthropic_config
from dheera_ai.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from dheera_ai.proxy._types import PassThroughEndpointLoggingTypedDict
from dheera_ai.proxy.auth.auth_utils import get_end_user_id_from_request_body
from dheera_ai.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from dheera_ai.types.utils import DheeraAIBatch, ModelResponse, TextCompletionResponse

if TYPE_CHECKING:
    from dheera_ai.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class AnthropicPassthroughLoggingHandler:
    @staticmethod
    def anthropic_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: DheeraAILoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Optional[dict] = None,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Transforms Anthropic response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        # Check if this is a batch creation request
        if "/v1/messages/batches" in url_route and httpx_response.status_code == 200:
            # Get request body from parameter or kwargs
            request_body = request_body or kwargs.get("request_body", {})
            return AnthropicPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
        
        model = response_body.get("model", "")
        anthropic_config = get_anthropic_config(url_route)
        dheera_ai_model_response: ModelResponse = anthropic_config().transform_response(
            raw_response=httpx_response,
            model_response=dheera_ai.ModelResponse(),
            model=model,
            messages=[],
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            request_data={},
            encoding=dheera_ai.encoding,
            json_mode=False,
            dheera_ai_params={},
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            dheera_ai_model_response=dheera_ai_model_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        return {
            "result": dheera_ai_model_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _get_user_from_metadata(
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return get_end_user_id_from_request_body(request_body)
        return None

    @staticmethod
    def _create_anthropic_response_logging_payload(
        dheera_ai_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: DheeraAILoggingObj,
    ):
        """
        Create the standard logging object for Anthropic passthrough

        handles streaming and non-streaming responses
        """
        try:
            # Get custom_llm_provider from logging object if available (e.g., azure_ai for Azure Anthropic)
            custom_llm_provider = logging_obj.model_call_details.get(
                "custom_llm_provider"
            )

            # Prepend custom_llm_provider to model if not already present
            model_for_cost = model
            if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
                model_for_cost = f"{custom_llm_provider}/{model}"

            response_cost = dheera_ai.completion_cost(
                completion_response=dheera_ai_model_response,
                model=model_for_cost,
                custom_llm_provider=custom_llm_provider,
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (  # type: ignore
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = AnthropicPassthroughLoggingHandler._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("dheera_ai_params", {})
                    kwargs["dheera_ai_params"].update(
                        {"proxy_server_request": {"body": {"user": user}}}
                    )

            # pretty print standard logging object
            verbose_proxy_logger.debug(
                "kwargs= %s",
                json.dumps(kwargs, indent=4, default=str),
            )

            # set dheera_ai_call_id to logging response object
            dheera_ai_model_response.id = logging_obj.dheera_ai_call_id
            dheera_ai_model_response.model = model
            logging_obj.model_call_details["model"] = model
            if not logging_obj.model_call_details.get("custom_llm_provider"):
                logging_obj.model_call_details["custom_llm_provider"] = (
                    dheera_ai.LlmProviders.ANTHROPIC.value
                )
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error creating Anthropic response logging payload: %s", e
            )
            return kwargs

    @staticmethod
    def _handle_logging_anthropic_collected_chunks(
        dheera_ai_logging_obj: DheeraAILoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Anthropic passthrough endpoint and logs them in dheera_ai callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in dheera_ai callbacks
        """

        model = request_body.get("model", "")
        # Check if it's available in the logging object
        if (
            not model
            and hasattr(dheera_ai_logging_obj, "model_call_details")
            and dheera_ai_logging_obj.model_call_details.get("model")
        ):
            model = cast(str, dheera_ai_logging_obj.model_call_details.get("model"))

        complete_streaming_response = (
            AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                dheera_ai_logging_obj=dheera_ai_logging_obj,
                model=model,
            )
        )
        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Anthropic passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": {},
            }
        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            dheera_ai_model_response=complete_streaming_response,
            model=model,
            kwargs={},
            start_time=start_time,
            end_time=end_time,
            logging_obj=dheera_ai_logging_obj,
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _split_sse_chunk_into_events(chunk: Union[str, bytes]) -> List[str]:
        """
        Split a chunk that may contain multiple SSE events into individual events.

        SSE format: "event: type\ndata: {...}\n\n"
        Multiple events in a single chunk are separated by double newlines.

        Args:
            chunk: Raw chunk string that may contain multiple SSE events

        Returns:
            List of individual SSE event strings (each containing "event: X\ndata: {...}")
        """
        # Handle bytes input
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")

        # Split on double newlines to separate SSE events
        # Filter out empty strings
        events = [event.strip() for event in chunk.split("\n\n") if event.strip()]

        return events

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: Sequence[Union[str, bytes]],
        dheera_ai_logging_obj: DheeraAILoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw Anthropic chunks

        - Splits multi-event chunks into individual SSE events
        - Converts str chunks to generic chunks
        - Converts generic chunks to dheera_ai chunks (OpenAI format)
        - Builds complete response from dheera_ai chunks
        """
        verbose_proxy_logger.debug(
            "Building complete streaming response from %d chunks", len(all_chunks)
        )
        anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        all_openai_chunks = []

        # Process each chunk - a chunk may contain multiple SSE events
        for _chunk_str in all_chunks:
            # Split chunk into individual SSE events
            individual_events = (
                AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(
                    _chunk_str
                )
            )

            # Process each individual event
            for event_str in individual_events:
                try:
                    transformed_openai_chunk = anthropic_model_response_iterator.convert_str_chunk_to_generic_chunk(
                        chunk=event_str
                    )
                    if transformed_openai_chunk is not None:
                        all_openai_chunks.append(transformed_openai_chunk)

                except (StopIteration, StopAsyncIteration):
                    break

        complete_streaming_response = dheera_ai.stream_chunk_builder(
            chunks=all_openai_chunks,
            logging_obj=dheera_ai_logging_obj,
        )
        verbose_proxy_logger.debug(
            "Complete streaming response built: %s", complete_streaming_response
        )
        return complete_streaming_response

    @staticmethod
    def batch_creation_handler(  # noqa: PLR0915
        httpx_response: httpx.Response,
        logging_obj: DheeraAILoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Optional[dict] = None,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle Anthropic batch creation passthrough logging.
        Creates a managed object for cost tracking when batch job is successfully created.
        """
        import base64

        from dheera_ai._uuid import uuid
        from dheera_ai.llms.anthropic.batches.transformation import (
            AnthropicBatchesConfig,
        )
        from dheera_ai.types.utils import Choices, SpecialEnums

        try:
            _json_response = httpx_response.json()
            
            
            # Only handle successful batch job creation (POST requests with 201 status)
            if httpx_response.status_code == 200 and "id" in _json_response:
                # Transform Anthropic response to DheeraAI batch format
                anthropic_batches_config = AnthropicBatchesConfig()
                dheera_ai_batch_response = anthropic_batches_config.transform_retrieve_batch_response(
                    model=None,
                    raw_response=httpx_response,
                    logging_obj=logging_obj,
                    dheera_ai_params={},
                )
                # Set status to "validating" for newly created batches so polling mechanism picks them up
                # The polling mechanism only looks for status="validating" jobs
                dheera_ai_batch_response.status = "validating"
                
                # Extract batch ID from the response
                batch_id = _json_response.get("id", "")
                
                # Get model from request body (batch response doesn't include model)
                request_body = request_body or {}
                # Try to extract model from the batch request body, supporting Anthropic's nested structure
                model_name: str = "unknown"
                if isinstance(request_body, dict):
                    # Standard: {"model": ...}
                    model_name = request_body.get("model") or "unknown"
                    if model_name == "unknown":
                        # Anthropic batches: look under requests[0].params.model
                        requests_list = request_body.get("requests", [])
                        if isinstance(requests_list, list) and len(requests_list) > 0:
                            first_req = requests_list[0]
                            if isinstance(first_req, dict):
                                params = first_req.get("params", {})
                                if isinstance(params, dict):
                                    extracted_model = params.get("model")
                                    if extracted_model:
                                        model_name = extracted_model
                
                
                # Create unified object ID for tracking
                # Format: base64(dheera_ai_proxy;model_id:{};llm_batch_id:{})
                # For Anthropic passthrough, prefix model with "anthropic/" so router can determine provider
                actual_model_id = AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router(model_name)
                
                # If model not in router, use "anthropic/{model_name}" format so router can determine provider
                if actual_model_id == model_name and not actual_model_id.startswith("anthropic/"):
                    actual_model_id = f"anthropic/{model_name}"

                unified_id_string = SpecialEnums.DHEERA_AI_MANAGED_BATCH_COMPLETE_STR.value.format(actual_model_id, batch_id)
                unified_object_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
                
                # Store the managed object for cost tracking
                # This will be picked up by check_batch_cost polling mechanism
                AnthropicPassthroughLoggingHandler._store_batch_managed_object(
                    unified_object_id=unified_object_id,
                    batch_object=dheera_ai_batch_response,
                    model_object_id=batch_id,
                    logging_obj=logging_obj,
                    **kwargs,
                )
                
                # Create a batch job response for logging
                dheera_ai_model_response = ModelResponse()
                dheera_ai_model_response.id = str(uuid.uuid4())
                dheera_ai_model_response.model = model_name
                dheera_ai_model_response.object = "batch"
                dheera_ai_model_response.created = int(start_time.timestamp())
                
                # Add batch-specific metadata to indicate this is a pending batch job
                dheera_ai_model_response.choices = [Choices(
                    finish_reason="batch_pending",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch job {batch_id} created and is pending. Status will be updated when the batch completes.",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_id": batch_id,
                            "batch_job_state": "in_progress",
                            "unified_object_id": unified_object_id
                        }
                    }
                )]
                
                # Set response cost to 0 initially (will be updated when batch completes)
                response_cost = 0.0
                kwargs["response_cost"] = response_cost
                kwargs["model"] = model_name
                kwargs["batch_id"] = batch_id
                kwargs["unified_object_id"] = unified_object_id
                kwargs["batch_job_state"] = "in_progress"
                
                logging_obj.model = model_name
                logging_obj.model_call_details["model"] = logging_obj.model
                logging_obj.model_call_details["response_cost"] = response_cost
                logging_obj.model_call_details["batch_id"] = batch_id
                
                return {
                    "result": dheera_ai_model_response,
                    "kwargs": kwargs,
                }
            else:
                # Handle non-successful responses
                dheera_ai_model_response = ModelResponse()
                dheera_ai_model_response.id = str(uuid.uuid4())
                dheera_ai_model_response.model = "anthropic_batch"
                dheera_ai_model_response.object = "batch"
                dheera_ai_model_response.created = int(start_time.timestamp())
                
                # Add error-specific metadata
                dheera_ai_model_response.choices = [Choices(
                    finish_reason="batch_error",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch job creation failed. Status: {httpx_response.status_code}",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_state": "failed",
                            "status_code": httpx_response.status_code
                        }
                    }
                )]
                
                kwargs["response_cost"] = 0.0
                kwargs["model"] = "anthropic_batch"
                kwargs["batch_job_state"] = "failed"
                
                return {
                    "result": dheera_ai_model_response,
                    "kwargs": kwargs,
                }
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error in batch_creation_handler: {e}")
            # Return basic response on error
            dheera_ai_model_response = ModelResponse()
            dheera_ai_model_response.id = str(uuid.uuid4())
            dheera_ai_model_response.model = "anthropic_batch"
            dheera_ai_model_response.object = "batch"
            dheera_ai_model_response.created = int(start_time.timestamp())
            
            # Add error-specific metadata
            dheera_ai_model_response.choices = [Choices(
                finish_reason="batch_error",
                index=0,
                message={
                    "role": "assistant",
                    "content": f"Error creating batch job: {str(e)}",
                    "tool_calls": None,
                    "function_call": None,
                    "provider_specific_fields": {
                        "batch_job_state": "failed",
                        "error": str(e)
                    }
                }
            )]
            
            kwargs["response_cost"] = 0.0
            kwargs["model"] = "anthropic_batch"
            kwargs["batch_job_state"] = "failed"
            
            return {
                "result": dheera_ai_model_response,
                "kwargs": kwargs,
            }

    @staticmethod
    def _store_batch_managed_object(
        unified_object_id: str,
        batch_object: DheeraAIBatch,
        model_object_id: str,
        logging_obj: DheeraAILoggingObj,
        **kwargs,
    ) -> None:
        """
        Store batch managed object for cost tracking.
        This will be picked up by the check_batch_cost polling mechanism.
        """
        try:
            
            # Get the managed files hook from the logging object
            # This is a bit of a hack, but we need access to the proxy logging system
            from dheera_ai.proxy.proxy_server import proxy_logging_obj
            
            managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_hook is not None and hasattr(managed_files_hook, 'store_unified_object_id'):
                # Create a mock user API key dict for the managed object storage
                from dheera_ai.proxy._types import LitellmUserRoles, UserAPIKeyAuth
                user_api_key_dict = UserAPIKeyAuth(
                    user_id=kwargs.get("user_id", "default-user"),
                    api_key="",
                    team_id=None,
                    team_alias=None,
                    user_role=LitellmUserRoles.CUSTOMER,  # Use proper enum value
                    user_email=None,
                    max_budget=None,
                    spend=0.0,  # Set to 0.0 instead of None
                    models=[],  # Set to empty list instead of None
                    tpm_limit=None,
                    rpm_limit=None,
                    budget_duration=None,
                    budget_reset_at=None,
                    max_parallel_requests=None,
                    allowed_model_region=None,
                    metadata={},  # Set to empty dict instead of None
                    key_alias=None,
                    permissions={},  # Set to empty dict instead of None
                    model_max_budget={},  # Set to empty dict instead of None
                    model_spend={},  # Set to empty dict instead of None
                )
                
                # Store the unified object for batch cost tracking
                import asyncio
                asyncio.create_task(
                    managed_files_hook.store_unified_object_id(  # type: ignore
                        unified_object_id=unified_object_id,
                        file_object=batch_object,
                        dheera_ai_parent_otel_span=None,
                        model_object_id=model_object_id,
                        file_purpose="batch",
                        user_api_key_dict=user_api_key_dict,
                    )
                )
                
                verbose_proxy_logger.info(
                    f"Stored Anthropic batch managed object with unified_object_id={unified_object_id}, batch_id={model_object_id}"
                )
            else:
                verbose_proxy_logger.warning("Managed files hook not available, cannot store batch object for cost tracking")
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error storing Anthropic batch managed object: {e}")

    @staticmethod
    def get_actual_model_id_from_router(model_name: str) -> str:
        from dheera_ai.proxy.proxy_server import llm_router
        
        if llm_router is not None:
            # Try to find the model in the router by the model name
            # Use the existing get_model_ids method from router
            model_ids = llm_router.get_model_ids(model_name=model_name)
            if model_ids and len(model_ids) > 0:
                # Use the first model ID found
                actual_model_id = model_ids[0]
                verbose_proxy_logger.info(f"Found model ID in router: {actual_model_id}")
                return actual_model_id
            else:
                # Fallback to model name
                actual_model_id = model_name
                verbose_proxy_logger.warning(f"Model not found in router, using model name: {actual_model_id}")
                return actual_model_id
        else:
            # Fallback if router is not available
            verbose_proxy_logger.warning(f"Router not available, using model name: {model_name}")
            return model_name
