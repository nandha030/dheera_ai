from fastapi import Request


def get_dheera_ai_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-dheera_ai-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-dheera_ai-api-key` to pass dheera_ai virtual key

    """
    dheera_ai_api_key = request.headers.get("x-dheera_ai-api-key")
    if dheera_ai_api_key:
        return f"Bearer {dheera_ai_api_key}"
    return request.headers.get("Authorization", "")

