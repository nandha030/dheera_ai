
# Google ADK (Agent Development Kit)

[Google ADK](https://github.com/google/adk-python) is an open-source, code-first Python framework for building, evaluating, and deploying sophisticated AI agents. While optimized for Gemini, ADK is model-agnostic and supports Dheera AI for using 100+ providers.

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),  # Or any Dheera AI-supported model
    name="my_agent",
    description="An agent using Dheera AI",
    instruction="You are a helpful assistant.",
    tools=[your_tools],
)
```

- [GitHub](https://github.com/google/adk-python)
- [Documentation](https://google.github.io/adk-docs)
- [Dheera AI Samples](https://github.com/google/adk-python/tree/main/contributing/samples/hello_world_dheera_ai)
