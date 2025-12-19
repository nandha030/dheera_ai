"""Test health check helper functions"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from dheera_ai.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from dheera_ai.dheera_ai_core_utils.health_check_helpers import HealthCheckHelpers
from dheera_ai.proxy._types import UserAPIKeyAuth


def test_update_model_params_with_health_check_tracking_information():
    """Test _update_model_params_with_health_check_tracking_information adds required tracking info."""
    initial_model_params = {
        "model": "gpt-3.5-turbo",
        "api_key": "test_key"
    }
    
    with patch(
        "dheera_ai.proxy._types.UserAPIKeyAuth.get_dheera_ai_internal_health_check_user_api_key_auth"
    ) as mock_get_auth:
        mock_auth = MagicMock()
        mock_get_auth.return_value = mock_auth
        
        with patch(
            "dheera_ai.proxy.dheera_ai_pre_call_utils.DheeraAIProxyRequestSetup.add_user_api_key_auth_to_request_metadata"
        ) as mock_add_auth:
            mock_add_auth.return_value = {
                **initial_model_params,
                "dheera_ai_metadata": {
                    "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
                    "user_api_key_auth": mock_auth
                }
            }
            
            result = HealthCheckHelpers._update_model_params_with_health_check_tracking_information(
                initial_model_params
            )
            
            # Verify that dheera_ai_metadata was added
            assert "dheera_ai_metadata" in result
            assert result["dheera_ai_metadata"]["tags"] == [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]
            
            # Verify the auth setup was called
            mock_add_auth.assert_called_once()
            call_args = mock_add_auth.call_args
            assert call_args[1]["user_api_key_dict"] == mock_auth
            assert call_args[1]["_metadata_variable_name"] == "dheera_ai_metadata"


def test_get_metadata_for_health_check_call():
    """Test _get_metadata_for_health_check_call returns correct metadata structure."""
    result = HealthCheckHelpers._get_metadata_for_health_check_call()
    
    expected_metadata = {
        "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
    }
    
    assert result == expected_metadata
    assert isinstance(result["tags"], list)
    assert len(result["tags"]) == 1
    assert result["tags"][0] == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


def test_get_dheera_ai_internal_health_check_user_api_key_auth():
    """Test get_dheera_ai_internal_health_check_user_api_key_auth returns properly configured UserAPIKeyAuth object."""
    result = UserAPIKeyAuth.get_dheera_ai_internal_health_check_user_api_key_auth()
    
    # Verify the returned object is of correct type
    assert isinstance(result, UserAPIKeyAuth)
    
    # Verify all fields are set to the expected constant value
    assert result.api_key == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_id == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.key_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME 