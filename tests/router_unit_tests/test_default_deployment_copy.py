"""
Regression test for default_deployment shallow copy optimization.

Tests the critical side effect: ensure modifying returned deployment
doesn't corrupt the original default_deployment instance.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from dheera_ai import Router


def test_default_deployment_isolation():
    """
    Regression test for shallow copy optimization in _common_checks_available_deployment.
    
    When a model is not in model_names and default_deployment is set, the router
    returns a copy of default_deployment with the model name updated. This test
    ensures the optimization (shallow copy instead of deepcopy) properly isolates
    each returned deployment from the original and from each other.
    
    The shallow copy optimization copies two levels:
    1. Top-level deployment dict
    2. dheera_ai_params dict
    
    Deeper nested objects are intentionally shared for performance (safe because
    the router only modifies the 'model' field at dheera_ai_params level).
    
    Critical behavior verified:
    1. Each deployment gets independent model value
    2. Original default_deployment unchanged for dheera_ai_params fields
    3. Shared fields (api_key) accessible in all copies
    4. Adding new dheera_ai_params fields is isolated per deployment
    5. Deep nested objects ARE shared (acceptable trade-off)
    """
    # Setup: Router with a default deployment (used for unknown models)
    router = Router(model_list=[])
    
    router.default_deployment = {  # type: ignore
        "model_name": "default-model",
        "dheera_ai_params": {
            "model": "gpt-3.5-turbo",  # This will be overwritten per request
            "api_key": "test-key",      # This should be shared
            "custom_config": {          # Deep nested - will be SHARED
                "nested_setting": "original",
            },
        },
    }
    
    # Act: Request two different unknown models (triggers default deployment path)
    _, deployment1 = router._common_checks_available_deployment(
        model="custom-model-1",  # Unknown model
        messages=[{"role": "user", "content": "test"}],
    )
    
    _, deployment2 = router._common_checks_available_deployment(
        model="custom-model-2",  # Different unknown model
        messages=[{"role": "user", "content": "test"}],
    )
    
    # Assert: Each deployment should have its own independent model value
    assert deployment1["dheera_ai_params"]["model"] == "custom-model-1"  # type: ignore
    assert deployment2["dheera_ai_params"]["model"] == "custom-model-2"  # type: ignore
    
    # Assert: Original default_deployment must remain unchanged (not mutated by requests)
    assert router.default_deployment["dheera_ai_params"]["model"] == "gpt-3.5-turbo"  # type: ignore
    
    # Assert: Shared fields should still be accessible in all copies
    assert deployment1["dheera_ai_params"]["api_key"] == "test-key"  # type: ignore
    assert deployment2["dheera_ai_params"]["api_key"] == "test-key"  # type: ignore
    
    # Assert: Modifying dheera_ai_params in one deployment doesn't affect others
    # This tests the shallow copy properly isolated the dheera_ai_params dict level
    deployment1["dheera_ai_params"]["temperature"] = 0.9  # type: ignore
    assert "temperature" not in deployment2["dheera_ai_params"]  # type: ignore
    assert "temperature" not in router.default_deployment["dheera_ai_params"]  # type: ignore
    
    # Assert: Deep nested objects ARE shared (intentional trade-off for 100x perf gain)
    # Safe because router only modifies top-level dheera_ai_params fields
    deployment1["dheera_ai_params"]["custom_config"]["nested_setting"] = "modified"  # type: ignore
    assert deployment2["dheera_ai_params"]["custom_config"]["nested_setting"] == "modified"  # type: ignore
    assert router.default_deployment["dheera_ai_params"]["custom_config"]["nested_setting"] == "modified"  # type: ignore

