import os
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)

from dheera_ai.proxy.discovery_endpoints.ui_discovery_endpoints import router


def test_ui_discovery_endpoints_with_defaults():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {}, clear=False):
        
        response = client.get("/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False


def test_ui_discovery_endpoints_with_custom_server_root_path():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/dheera_ai"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {}, clear=False):
        
        response = client.get("/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/dheera_ai"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False


def test_ui_discovery_endpoints_with_proxy_base_url_when_set():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {}, clear=False):
        
        response = client.get("/dheera_ai/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is False


def test_ui_discovery_endpoints_with_sso_configured_and_auto_redirect_enabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/dheera_ai"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true"}, clear=False):
        
        response = client.get("/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/dheera_ai"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is True


def test_ui_discovery_endpoints_with_sso_configured_but_auto_redirect_disabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/dheera_ai"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "false"}, clear=False):
        
        response = client.get("/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/dheera_ai"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is False


def test_ui_discovery_endpoints_with_sso_not_configured_but_auto_redirect_enabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true"}, clear=False):
        
        response = client.get("/.well-known/dheera_ai-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False


def test_ui_discovery_endpoints_both_routes_return_same_data():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("dheera_ai.proxy.utils.get_server_root_path", return_value="/dheera_ai"), \
         patch("dheera_ai.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("dheera_ai.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true"}, clear=False):
        
        response1 = client.get("/.well-known/dheera_ai-ui-config")
        response2 = client.get("/dheera_ai/.well-known/dheera_ai-ui-config")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()

