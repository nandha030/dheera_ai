"""
Cached imports module for DheeraAI.

This module provides cached import functionality to avoid repeated imports
inside functions that are critical to performance.
"""

from typing import TYPE_CHECKING, Callable, Optional, Type

# Type annotations for cached imports
if TYPE_CHECKING:
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging
    from dheera_ai.dheera_ai_core_utils.coroutine_checker import CoroutineChecker

# Global cache variables
_DheeraAILogging: Optional[Type["Logging"]] = None
_coroutine_checker: Optional["CoroutineChecker"] = None
_set_callbacks: Optional[Callable] = None


def get_dheera_ai_logging_class() -> Type["Logging"]:
    """Get the cached DheeraAI Logging class, initializing if needed."""
    global _DheeraAILogging
    if _DheeraAILogging is not None:
        return _DheeraAILogging
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import Logging
    _DheeraAILogging = Logging
    return _DheeraAILogging


def get_coroutine_checker() -> "CoroutineChecker":
    """Get the cached coroutine checker instance, initializing if needed."""
    global _coroutine_checker
    if _coroutine_checker is not None:
        return _coroutine_checker
    from dheera_ai.dheera_ai_core_utils.coroutine_checker import coroutine_checker
    _coroutine_checker = coroutine_checker
    return _coroutine_checker


def get_set_callbacks() -> Callable:
    """Get the cached set_callbacks function, initializing if needed."""
    global _set_callbacks
    if _set_callbacks is not None:
        return _set_callbacks
    from dheera_ai.dheera_ai_core_utils.dheera_ai_logging import set_callbacks
    _set_callbacks = set_callbacks
    return _set_callbacks


def clear_cached_imports() -> None:
    """Clear all cached imports. Useful for testing or memory management."""
    global _DheeraAILogging, _coroutine_checker, _set_callbacks
    _DheeraAILogging = None
    _coroutine_checker = None
    _set_callbacks = None
