"""
Example Custom SSO Handler

Use this if you want to run custom code after dheera_ai has retrieved information from your IDP (Identity Provider).

Flow:
- User lands on Admin UI
- DheeraAI redirects user to your SSO provider
- Your SSO provider redirects user back to DheeraAI
- DheeraAI has retrieved user information from your IDP
- Your custom SSO handler is called and returns an object of type SSOUserDefinedValues
- User signed in to UI
"""

from fastapi_sso.sso.base import OpenID

from dheera_ai.proxy._types import LitellmUserRoles, SSOUserDefinedValues
from dheera_ai.proxy.management_endpoints.internal_user_endpoints import user_info


async def custom_sso_handler(userIDPInfo: OpenID) -> SSOUserDefinedValues:
    try:
        print("inside custom sso handler")  # noqa
        print(f"userIDPInfo: {userIDPInfo}")  # noqa

        if userIDPInfo.id is None:
            raise ValueError(
                f"No ID found for user. userIDPInfo.id is None {userIDPInfo}"
            )

        # check if user exists in dheera_ai proxy DB
        _user_info = await user_info(user_id=userIDPInfo.id)
        print("_user_info from dheera_ai DB ", _user_info)  # noqa

        return SSOUserDefinedValues(
            models=[],
            user_id=userIDPInfo.id,
            user_email=userIDPInfo.email,
            user_role=LitellmUserRoles.INTERNAL_USER.value,
            max_budget=10,
            budget_duration="1d",
        )
    except Exception:
        raise Exception("Failed custom auth")
