"""DheeraAI SDK functions for managing vector store files."""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import dheera_ai
from dheera_ai.constants import request_timeout
from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging as DheeraAILoggingObj
from dheera_ai.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from dheera_ai.types.router import GenericDheeraAIParams
from dheera_ai.types.utils import LlmProviders
from dheera_ai.types.vector_store_files import (
    VectorStoreFileContentResponse,
    VectorStoreFileCreateRequest,
    VectorStoreFileDeleteResponse,
    VectorStoreFileListQueryParams,
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileUpdateRequest,
)
from dheera_ai.utils import ProviderConfigManager, client
from dheera_ai.vector_store_files.utils import VectorStoreFileRequestUtils

base_llm_http_handler = BaseLLMHTTPHandler()

VectorStoreFileAttributeValue = Union[str, int, float, bool]
VectorStoreFileAttributes = Dict[str, VectorStoreFileAttributeValue]


def _ensure_provider(custom_llm_provider: Optional[str]) -> str:
    return custom_llm_provider or "openai"


def _prepare_registry_credentials(
    *,
    vector_store_id: str,
    kwargs: Dict[str, Any],
) -> None:
    if dheera_ai.vector_store_registry is None:
        return
    try:
        registry_credentials = (
            dheera_ai.vector_store_registry.get_credentials_for_vector_store(
                vector_store_id
            )
        )
        if registry_credentials:
            kwargs.update(registry_credentials)
    except Exception:
        pass


@client
async def acreate(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: Optional[VectorStoreFileAttributes] = None,
    chunking_strategy: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate"] = True

        func = partial(
            create,
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: Optional[VectorStoreFileAttributes] = None,
    chunking_strategy: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("acreate", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file create is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)
        create_request: VectorStoreFileCreateRequest = (
            VectorStoreFileRequestUtils.get_create_request_params(local_vars)
        )
        create_request["file_id"] = file_id

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                **create_request,
            },
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_create_handler(
            vector_store_id=vector_store_id,
            create_request=create_request,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist(
    *,
    vector_store_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    filter: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileListResponse:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist"] = True

        func = partial(
            list,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list(
    *,
    vector_store_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    filter: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreFileListResponse, Coroutine[Any, Any, VectorStoreFileListResponse]]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("alist", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file list is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)
        list_query: VectorStoreFileListQueryParams = (
            VectorStoreFileRequestUtils.get_list_query_params(local_vars)
        )

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={"vector_store_id": vector_store_id, **list_query},
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_list_handler(
            vector_store_id=vector_store_id,
            query_params=list_query,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aretrieve(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aretrieve"] = True

        func = partial(
            retrieve,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def retrieve(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("aretrieve", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file retrieve is not supported for {custom_llm_provider}"
            )

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_retrieve_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aretrieve_content(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileContentResponse:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aretrieve_content"] = True

        func = partial(
            retrieve_content,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def retrieve_content(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    VectorStoreFileContentResponse, Coroutine[Any, Any, VectorStoreFileContentResponse]
]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("aretrieve_content", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file content retrieve is not supported for {custom_llm_provider}"
            )

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_content_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aupdate(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aupdate"] = True

        func = partial(
            update,
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def update(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("aupdate", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file update is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)
        update_request: VectorStoreFileUpdateRequest = (
            VectorStoreFileRequestUtils.get_update_request_params(local_vars)
        )
        update_request["attributes"] = attributes

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
                **update_request,
            },
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_update_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            update_request=update_request,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def adelete(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreFileDeleteResponse:
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete"] = True

        func = partial(
            delete,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    VectorStoreFileDeleteResponse, Coroutine[Any, Any, VectorStoreFileDeleteResponse]
]:
    local_vars = locals()
    try:
        dheera_ai_logging_obj: DheeraAILoggingObj = kwargs.get("dheera_ai_logging_obj")  # type: ignore
        dheera_ai_call_id: Optional[str] = kwargs.get("dheera_ai_call_id")
        _is_async = kwargs.pop("adelete", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        dheera_ai_params = GenericDheeraAIParams(
            vector_store_id=vector_store_id, **kwargs
        )

        provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Vector store file delete is not supported for {custom_llm_provider}"
            )

        dheera_ai_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            dheera_ai_params={
                "vector_store_id": vector_store_id,
                "dheera_ai_call_id": dheera_ai_call_id,
                **dheera_ai_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_file_delete_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            dheera_ai_params=dheera_ai_params,
            logging_obj=dheera_ai_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise dheera_ai.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
