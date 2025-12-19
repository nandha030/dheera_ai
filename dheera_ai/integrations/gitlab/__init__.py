from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from .gitlab_prompt_manager import GitLabPromptManager
    from dheera_ai.types.prompts.init_prompts import PromptDheeraAIParams, PromptSpec
    from dheera_ai.integrations.custom_prompt_management import CustomPromptManagement

from dheera_ai.types.prompts.init_prompts import SupportedPromptIntegrations
from dheera_ai.integrations.custom_prompt_management import CustomPromptManagement
from dheera_ai.types.prompts.init_prompts import PromptSpec, PromptDheeraAIParams
from .gitlab_prompt_manager import GitLabPromptManager, GitLabPromptCache

# Global instances
global_gitlab_config: Optional[dict] = None


def set_global_gitlab_config(config: dict) -> None:
    """
    Set the global gitlab configuration for prompt management.

    Args:
        config: Dictionary containing gitlab configuration
                - workspace: gitlab workspace name
                - repository: Repository name
                - access_token: gitlab access token
                - branch: Branch to fetch prompts from (default: main)
    """
    import dheera_ai

    dheera_ai.global_gitlab_config = config  # type: ignore


def prompt_initializer(
    dheera_ai_params: "PromptDheeraAIParams", prompt_spec: "PromptSpec"
) -> "CustomPromptManagement":
    """
    Initialize a prompt from a Gitlab repository.
    """
    gitlab_config = getattr(dheera_ai_params, "gitlab_config", None)
    prompt_id = getattr(dheera_ai_params, "prompt_id", None)


    if not gitlab_config:
        raise ValueError(
            "gitlab_config is required for gitlab prompt integration"
        )

    try:
        gitlab_prompt_manager = GitLabPromptManager(
            gitlab_config=gitlab_config,
            prompt_id=prompt_id,
        )

        return gitlab_prompt_manager
    except Exception as e:
        raise e

def _gitlab_prompt_initializer(
        dheera_ai_params: PromptDheeraAIParams,
        prompt: PromptSpec,
) -> CustomPromptManagement:
    """
    Build a GitLab-backed prompt manager for this prompt.
    Expected fields on dheera_ai_params:
      - prompt_integration="gitlab"  (handled by the caller)
      - gitlab_config: Dict[str, Any] (project/access_token/branch/prompts_path/etc.)
      - git_ref (optional): per-prompt tag/branch/SHA override
    """
    # You can store arbitrary integration-specific config on PromptDheeraAIParams.
    # If your dataclass doesn't have these attributes, add them or put inside
    # `dheera_ai_params.extra` and pull them from there.
    gitlab_config: Dict[str, Any] = getattr(dheera_ai_params, "gitlab_config", None) or {}
    git_ref: Optional[str] = getattr(dheera_ai_params, "git_ref", None)

    if not gitlab_config:
        raise ValueError("gitlab_config is required for gitlab prompt integration")

    # prompt.prompt_id can map to a file path under prompts_path (e.g. "chat/greet/hi")
    return GitLabPromptManager(
        gitlab_config=gitlab_config,
        prompt_id=prompt.prompt_id,
        ref=git_ref,
    )


prompt_initializer_registry = {
    SupportedPromptIntegrations.GITLAB.value: _gitlab_prompt_initializer,
}

# Export public API
__all__ = [
    "GitLabPromptManager",
    "GitLabPromptCache",
    "set_global_gitlab_config",
    "global_gitlab_config",
]
