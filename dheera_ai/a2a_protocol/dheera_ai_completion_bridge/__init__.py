"""
A2A to DheeraAI Completion Bridge.

This module provides transformation between A2A protocol messages and
DheeraAI completion API, enabling any DheeraAI-supported provider to be
invoked via the A2A protocol.
"""

from dheera_ai.a2a_protocol.dheera_ai_completion_bridge.handler import (
    A2ACompletionBridgeHandler,
    handle_a2a_completion,
    handle_a2a_completion_streaming,
)
from dheera_ai.a2a_protocol.dheera_ai_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
)

__all__ = [
    "A2ACompletionBridgeTransformation",
    "A2ACompletionBridgeHandler",
    "handle_a2a_completion",
    "handle_a2a_completion_streaming",
]
