"""Utils for accessing credentials."""

from typing import List

import dheera_ai
from dheera_ai.types.utils import CredentialItem


class CredentialAccessor:
    @staticmethod
    def get_credential_values(credential_name: str) -> dict:
        """Safe accessor for credentials."""

        if not dheera_ai.credential_list:
            return {}
        for credential in dheera_ai.credential_list:
            if credential.credential_name == credential_name:
                return credential.credential_values.copy()
        return {}

    @staticmethod
    def upsert_credentials(credentials: List[CredentialItem]):
        """Add a credential to the list of credentials."""

        credential_names = [cred.credential_name for cred in dheera_ai.credential_list]

        for credential in credentials:
            if credential.credential_name in credential_names:
                # Find and replace the existing credential in the list
                for i, existing_cred in enumerate(dheera_ai.credential_list):
                    if existing_cred.credential_name == credential.credential_name:
                        dheera_ai.credential_list[i] = credential
                        break
            else:
                dheera_ai.credential_list.append(credential)
