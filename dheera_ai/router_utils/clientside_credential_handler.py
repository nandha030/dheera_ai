"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

clientside_credential_keys = ["api_key", "api_base", "base_url"]


def is_clientside_credential(request_kwargs: dict) -> bool:
    """
    Check if the credential is a clientside credential.
    """
    return any(key in request_kwargs for key in clientside_credential_keys)


def get_dynamic_dheera_ai_params(dheera_ai_params: dict, request_kwargs: dict) -> dict:
    """
    Generate a unique model_id for the deployment.

    Returns
    - dheera_ai_params: dict

    for generating a unique model_id.
    """
    # update dheera_ai_params with clientside credentials
    for key in clientside_credential_keys:
        if key in request_kwargs:
            dheera_ai_params[key] = request_kwargs[key]
    return dheera_ai_params
