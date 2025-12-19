
import asyncio
import os
import subprocess
import sys
import time
import traceback
import platform

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

def test_using_dheera_ai_on_windows():
    """Test that DheeraAI can be imported on Windows systems."""
    
    try:
        import dheera_ai
        print(f"dheera_ai imported successfully on Windows ({platform.system()} {platform.release()})")

        response = dheera_ai.completion(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "This should never fail. Email ishaan@berri.ai if this test ever fails."}
            ],
            mock_response="Hello, how are you?"
        )
        print(response)
    except Exception as e:
        pytest.fail(
            f"Error occurred on Windows: {e}. Installing dheera_ai on Windows failed."
        )

