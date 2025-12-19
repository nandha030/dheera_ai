import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# xAI

https://docs.x.ai/docs

:::tip

**We support ALL xAI models, just set `model=xai/<any-model-on-xai>` as a prefix when sending dheera_ai requests**

:::

## Supported Models



**Latest Release** - Grok 4.1 Fast: Optimized for high-performance agentic tool calling with 2M context and prompt caching.

| Model | Context | Features |
|-------|---------|----------|
| `xai/grok-4-1-fast-reasoning` | 2M tokens | **Reasoning**, Function calling, Vision, Audio, Web search, Caching |
| `xai/grok-4-1-fast-non-reasoning` | 2M tokens | Function calling, Vision, Audio, Web search, Caching |

**When to use:**
- ✅ **Reasoning model**: Complex analysis, planning, multi-step reasoning problems
- ✅ **Non-reasoning model**: Simple queries, faster responses, lower token usage

**Example:**
```python
from dheera_ai import completion

# With reasoning
response = completion(
    model="xai/grok-4-1-fast-reasoning",
    messages=[{"role": "user", "content": "Analyze this problem step by step..."}]
)

# Without reasoning
response = completion(
    model="xai/grok-4-1-fast-non-reasoning",
    messages=[{"role": "user", "content": "What's 2+2?"}]
)
```

---

### All Available Models

| Model Family | Model | Context | Features |
|--------------|-------|---------|----------|
| **Grok 4.1** | `xai/grok-4-1-fast-reasoning` | 2M | **Reasoning**, Tools, Vision, Audio, Web search, Caching |
| | `xai/grok-4-1-fast-non-reasoning` | 2M | Tools, Vision, Audio, Web search, Caching |
| **Grok 4** | `xai/grok-4` | 256K | Tools, Web search |
| | `xai/grok-4-0709` | 256K | Tools, Web search |
| | `xai/grok-4-fast-reasoning` | 2M | **Reasoning**, Tools, Web search |
| | `xai/grok-4-fast-non-reasoning` | 2M | Tools, Web search |
| **Grok 3** | `xai/grok-3` | 131K | Tools, Web search |
| | `xai/grok-3-mini` | 131K | Tools, Web search |
| | `xai/grok-3-fast-beta` | 131K | Tools, Web search |
| **Grok Code** | `xai/grok-code-fast` | 256K | **Reasoning**, Tools, Code generation, Caching |
| **Grok 2** | `xai/grok-2` | 131K | Tools, **Vision** |
| | `xai/grok-2-vision-latest` | 32K | Tools, **Vision** |

**Features:**
- **Reasoning** = Chain-of-thought reasoning with reasoning tokens
- **Tools** = Function calling / Tool use
- **Web search** = Live internet search
- **Vision** = Image understanding
- **Audio** = Audio input support
- **Caching** = Prompt caching for cost savings
- **Code generation** = Optimized for code tasks

**Pricing:** See [xAI's pricing page](https://docs.x.ai/docs/models) for current rates.

## API Key
```python
# env variable
os.environ['XAI_API_KEY']
```

## Sample Usage

```python showLineNumbers title="Dheera AI python sdk usage - Non-streaming"
from dheera_ai import completion
import os

os.environ['XAI_API_KEY'] = ""
response = completion(
    model="xai/grok-3-mini-beta",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)
print(response)
```

## Sample Usage - Streaming

```python showLineNumbers title="Dheera AI python sdk usage - Streaming"
from dheera_ai import completion
import os

os.environ['XAI_API_KEY'] = ""
response = completion(
    model="xai/grok-3-mini-beta",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Vision

```python showLineNumbers title="Dheera AI python sdk usage - Vision"
import os 
from dheera_ai import completion

os.environ["XAI_API_KEY"] = "your-api-key"

response = completion(
    model="xai/grok-2-vision-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
            ],
        },
    ],
)
```

## Usage with Dheera AI Proxy Server

Here's how to call a XAI model with the Dheera AI Proxy Server

1. Modify the config.yaml 

  ```yaml showLineNumbers
  model_list:
    - model_name: my-model
      dheera_ai_params:
        model: xai/<your-model-name>  # add xai/ prefix to route as XAI provider
        api_key: api-key                 # api key to send your model
  ```


2. Start the proxy 

  ```bash
  $ dheera_ai --config /path/to/config.yaml
  ```

3. Send Request to Dheera AI Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python showLineNumbers
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass dheera_ai proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # dheera_ai-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>


## Reasoning Usage

Dheera AI supports reasoning usage for xAI models.

<Tabs>

<TabItem value="python" label="Dheera AI Python SDK">

```python showLineNumbers title="reasoning with xai/grok-3-mini-beta"
import dheera_ai
response = dheera_ai.completion(
    model="xai/grok-3-mini-beta",
    messages=[{"role": "user", "content": "What is 101*3?"}],
    reasoning_effort="low",
)

print("Reasoning Content:")
print(response.choices[0].message.reasoning_content)

print("\nFinal Response:")
print(completion.choices[0].message.content)

print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)

print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)
```
</TabItem>

<TabItem value="curl" label="Dheera AI Proxy - OpenAI SDK Usage">

```python showLineNumbers title="reasoning with xai/grok-3-mini-beta"
import openai
client = openai.OpenAI(
    api_key="sk-1234",             # pass dheera_ai proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000" # dheera_ai-proxy-base url
)

response = client.chat.completions.create(
    model="xai/grok-3-mini-beta",
    messages=[{"role": "user", "content": "What is 101*3?"}],
    reasoning_effort="low",
)

print("Reasoning Content:")
print(response.choices[0].message.reasoning_content)

print("\nFinal Response:")
print(completion.choices[0].message.content)

print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)

print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)
```

</TabItem>
</Tabs>

**Example Response:**

```shell
Reasoning Content:
Let me calculate 101 multiplied by 3:
101 * 3 = 303.
I can double-check that: 100 * 3 is 300, and 1 * 3 is 3, so 300 + 3 = 303. Yes, that's correct.

Final Response:
The result of 101 multiplied by 3 is 303.

Number of completion tokens (input):
14

Number of reasoning tokens (input):
310
```
