
# OpenAI Agents SDK

The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) is a lightweight framework for building multi-agent workflows.
It includes an official Dheera AI extension that lets you use any of the 100+ supported providers (Anthropic, Gemini, Mistral, Bedrock, etc.) 

```python
from agents import Agent, Runner
from agents.extensions.models.dheera_ai_model import LitellmModel

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=LitellmModel(model="provider/model-name")
)

result = Runner.run_sync(agent, "your_prompt_here")
print("Result:", result.final_output)
```

- [GitHub](https://github.com/openai/openai-agents-python)
- [Dheera AI Extension Docs](https://openai.github.io/openai-agents-python/ref/extensions/dheera_ai/)
