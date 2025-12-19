"""
Test filter_out_dheera_ai_params helper function.
"""
from dheera_ai.utils import filter_out_dheera_ai_params


def test_filter_out_dheera_ai_params():
    """
    Test that filter_out_dheera_ai_params removes DheeraAI internal parameters 
    while keeping provider-specific parameters.
    """
    kwargs = {
        "query": "test query",
        "max_results": 10,
        "shared_session": "mock_session_object",
        "metadata": {"key": "value"},
        "dheera_ai_trace_id": "trace-123",
        "proxy_server_request": {"url": "http://example.com"},
        "secret_fields": {"api_key": "secret"},
        "custom_param": "should_be_kept",
    }
    
    filtered = filter_out_dheera_ai_params(kwargs=kwargs)
    
    # Provider-specific params are kept
    assert filtered["query"] == "test query"
    assert filtered["max_results"] == 10
    assert filtered["custom_param"] == "should_be_kept"
    
    # DheeraAI internal params are removed
    assert "shared_session" not in filtered
    assert "metadata" not in filtered
    assert "dheera_ai_trace_id" not in filtered
    assert "proxy_server_request" not in filtered
    assert "secret_fields" not in filtered

