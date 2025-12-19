"""
DheeraAI Interactions API

This module provides SDK methods for Google's Interactions API.

Usage:
    import dheera_ai
    
    # Create an interaction with a model
    response = dheera_ai.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how are you?"
    )
    
    # Create an interaction with an agent
    response = dheera_ai.interactions.create(
        agent="deep-research-pro-preview-12-2025",
        input="Research the current state of cancer research"
    )
    
    # Async version
    response = await dheera_ai.interactions.acreate(...)
    
    # Get an interaction
    response = dheera_ai.interactions.get(interaction_id="...")
    
    # Delete an interaction
    result = dheera_ai.interactions.delete(interaction_id="...")
    
    # Cancel an interaction
    result = dheera_ai.interactions.cancel(interaction_id="...")

Methods:
- create(): Sync create interaction
- acreate(): Async create interaction
- get(): Sync get interaction
- aget(): Async get interaction
- delete(): Sync delete interaction
- adelete(): Async delete interaction
- cancel(): Sync cancel interaction
- acancel(): Async cancel interaction
"""

from dheera_ai.interactions.main import (
    acancel,
    acreate,
    adelete,
    aget,
    cancel,
    create,
    delete,
    get,
)

__all__ = [
    # Create
    "create",
    "acreate",
    # Get
    "get",
    "aget",
    # Delete
    "delete",
    "adelete",
    # Cancel
    "cancel",
    "acancel",
]
