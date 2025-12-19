# A2A to Dheera AI Completion Bridge

Routes A2A protocol requests through `dheera_ai.acompletion`, enabling any Dheera AI-supported provider to be invoked via A2A.

## Flow

```
A2A Request → Transform → dheera_ai.acompletion → Transform → A2A Response
```

## SDK Usage

Use the existing `asend_message` and `asend_message_streaming` functions with `dheera_ai_params`:

```python
from dheera_ai.a2a_protocol import asend_message, asend_message_streaming
from a2a.types import SendMessageRequest, SendStreamingMessageRequest, MessageSendParams
from uuid import uuid4

# Non-streaming
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
    )
)
response = await asend_message(
    request=request,
    api_base="http://localhost:2024",
    dheera_ai_params={"custom_llm_provider": "langgraph", "model": "agent"},
)

# Streaming
stream_request = SendStreamingMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
    )
)
async for chunk in asend_message_streaming(
    request=stream_request,
    api_base="http://localhost:2024",
    dheera_ai_params={"custom_llm_provider": "langgraph", "model": "agent"},
):
    print(chunk)
```

## Proxy Usage

Configure an agent with `custom_llm_provider` in `dheera_ai_params`:

```yaml
agents:
  - agent_name: my-langgraph-agent
    agent_card_params:
      name: "LangGraph Agent"
      url: "http://localhost:2024"  # Used as api_base
    dheera_ai_params:
      custom_llm_provider: langgraph
      model: agent
```

When an A2A request hits `/a2a/{agent_id}/message/send`, the bridge:

1. Detects `custom_llm_provider` in agent's `dheera_ai_params`
2. Transforms A2A message → OpenAI messages
3. Calls `dheera_ai.acompletion(model="langgraph/agent", api_base="http://localhost:2024")`
4. Transforms response → A2A format

## Classes

- `A2ACompletionBridgeTransformation` - Static methods for message format conversion
- `A2ACompletionBridgeHandler` - Static methods for handling requests (streaming/non-streaming)

