"""
DheeraAI Search API Types

This module defines types for the unified search API across different providers.
"""
from typing import List, Optional

from typing_extensions import Required, TypedDict

from dheera_ai.types.utils import SearchProviders

# Re-export SearchProviders as SearchProvider for backwards compatibility
SearchProvider = SearchProviders

__all__ = ["SearchProvider", "SearchProviders"]



class SearchToolDheeraAIParams(TypedDict, total=False):
    """
    DheeraAI params for search tools configuration.
    """
    search_provider: Required[str]
    api_key: Optional[str]
    api_base: Optional[str]
    timeout: Optional[float]
    max_retries: Optional[int]


class SearchTool(TypedDict, total=False):
    """
    Search tool configuration.
    
    Example:
        {
            "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
            "search_tool_name": "dheera_ai-search",
            "dheera_ai_params": {
                "search_provider": "perplexity",
                "api_key": "sk-..."
            },
            "search_tool_info": {
                "description": "Perplexity search tool"
            }
        }
    """
    search_tool_id: Optional[str]
    search_tool_name: Required[str]
    dheera_ai_params: Required[SearchToolDheeraAIParams]
    search_tool_info: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]


class SearchToolInfoResponse(TypedDict, total=False):
    """Response model for search tool information."""
    search_tool_id: Optional[str]
    search_tool_name: str
    dheera_ai_params: dict
    search_tool_info: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]


class ListSearchToolsResponse(TypedDict):
    """Response model for listing search tools."""
    search_tools: List[SearchToolInfoResponse]


class AvailableSearchProvider(TypedDict):
    """Information about an available search provider."""
    provider_name: str
    ui_friendly_name: str


