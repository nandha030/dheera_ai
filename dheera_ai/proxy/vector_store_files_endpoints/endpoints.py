from typing import Dict, Optional

import dheera_ai
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from dheera_ai.proxy._types import UserAPIKeyAuth
from dheera_ai.proxy.auth.user_api_key_auth import user_api_key_auth
from dheera_ai.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from dheera_ai.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)
from dheera_ai.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_files_endpoint,
)
from dheera_ai.types.utils import LlmProviders

router = APIRouter()


def _update_request_data_with_dheera_ai_managed_vector_store_registry(
    data: Dict,
    vector_store_id: str,
) -> Dict:
    if dheera_ai.vector_store_registry is not None:
        vector_store_to_run = (
            dheera_ai.vector_store_registry.get_dheera_ai_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
        )
        if vector_store_to_run is not None:
            if "custom_llm_provider" in vector_store_to_run:
                data["custom_llm_provider"] = vector_store_to_run.get(
                    "custom_llm_provider"
                )
            if "dheera_ai_credential_name" in vector_store_to_run:
                data["dheera_ai_credential_name"] = vector_store_to_run.get(
                    "dheera_ai_credential_name"
                )
            if "dheera_ai_params" in vector_store_to_run:
                dheera_ai_params = vector_store_to_run.get("dheera_ai_params", {}) or {}
                data.update(dheera_ai_params)
    return data


async def _resolve_provider(
    *,
    data: Dict,
    request: Request,
) -> Optional[LlmProviders]:
    provider = (
        data.get("custom_llm_provider")
        or get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
    )

    if provider is None and request.method in {"POST", "PUT", "PATCH"}:
        provider = await get_custom_llm_provider_from_request_body(request=request)

    if provider is None:
        provider = "openai"

    try:
        return LlmProviders(provider)
    except Exception:
        return None


def _maybe_check_permissions(
    *,
    provider: Optional[LlmProviders],
    vector_store_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    if provider is None:
        return
    metadata = user_api_key_dict.metadata or {}
    team_metadata = user_api_key_dict.team_metadata or {}
    if not metadata.get("allowed_vector_store_indexes") and not team_metadata.get(
        "allowed_vector_store_indexes"
    ):
        return
    is_allowed_to_call_vector_store_files_endpoint(
        provider=provider,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )


@router.post(
    "/v1/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.post(
    "/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_create(
    vector_store_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    if "vector_store_id" not in data:
        data["vector_store_id"] = vector_store_id

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_create",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_list(
    vector_store_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    query_params = dict(request.query_params)
    data: Dict[str, Optional[str]] = {"vector_store_id": vector_store_id}
    data.update(query_params)

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_list",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_retrieve(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_retrieve",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_content(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_content",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.post(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_update(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    data["vector_store_id"] = vector_store_id
    data["file_id"] = file_id

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_update",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.delete(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_delete(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from dheera_ai.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_dheera_ai_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_delete",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
