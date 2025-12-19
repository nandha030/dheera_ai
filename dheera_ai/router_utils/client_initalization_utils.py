import asyncio
from typing import TYPE_CHECKING, Any

from dheera_ai.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from dheera_ai.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class InitalizeCachedClient:
    @staticmethod
    def set_max_parallel_requests_client(
        dheera_ai_router_instance: LitellmRouter, model: dict
    ):
        dheera_ai_params = model.get("dheera_ai_params", {})
        model_id = model["model_info"]["id"]
        rpm = dheera_ai_params.get("rpm", None)
        tpm = dheera_ai_params.get("tpm", None)
        max_parallel_requests = dheera_ai_params.get("max_parallel_requests", None)
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=dheera_ai_router_instance.default_max_parallel_requests,
        )
        if calculated_max_parallel_requests:
            semaphore = asyncio.Semaphore(calculated_max_parallel_requests)
            cache_key = f"{model_id}_max_parallel_requests_client"
            dheera_ai_router_instance.cache.set_cache(
                key=cache_key,
                value=semaphore,
                local_only=True,
            )
