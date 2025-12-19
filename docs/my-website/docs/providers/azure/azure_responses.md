import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Responses API

| Property | Details |
|-------|-------|
| Description | Azure OpenAI Responses API |
| `custom_llm_provider` on Dheera AI | `azure/` |
| Supported Operations | `/v1/responses`|
| Azure OpenAI Responses API | [Azure OpenAI Responses API ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses?tabs=python-secure) |
| Cost Tracking, Logging Support | ✅ Dheera AI will log, track cost for Responses API Requests |
| Supported OpenAI Params | ✅ All OpenAI params are supported, [See here](https://github.com/BerriAI/dheera_ai/blob/0717369ae6969882d149933da48eeb8ab0e691bd/dheera_ai/llms/openai/responses/transformation.py#L23) |

## Usage

## Create a model response

<Tabs>
<TabItem value="dheera_ai-sdk" label="Dheera AI SDK">

#### Non-streaming

```python showLineNumbers title="Azure Responses API"
import dheera_ai

# Non-streaming response
response = dheera_ai.responses(
    model="azure/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://dheera_ai8397336933.openai.azure.com/",
    api_version="2023-03-15-preview",
)

print(response)
```

#### Streaming
```python showLineNumbers title="Azure Responses API"
import dheera_ai

# Streaming response
response = dheera_ai.responses(
    model="azure/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://dheera_ai8397336933.openai.azure.com/",
    api_version="2023-03-15-preview",
)

for event in response:
    print(event)
```

</TabItem>
<TabItem value="proxy" label="OpenAI SDK with Dheera AI Proxy">

First, add this to your dheera_ai proxy config.yaml:
```yaml showLineNumbers title="Azure Responses API"
model_list:
  - model_name: o1-pro
    dheera_ai_params:
      model: azure/o1-pro
      api_key: os.environ/AZURE_RESPONSES_OPENAI_API_KEY
      api_base: https://dheera_ai8397336933.openai.azure.com/
      api_version: 2023-03-15-preview
```

Start your Dheera AI proxy:
```bash
dheera_ai --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Then use the OpenAI SDK pointed to your proxy:

#### Non-streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>
</Tabs>

## Azure Codex Models

Codex models use Azure's new [/v1/preview API](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-lifecycle?tabs=key#next-generation-api) which provides ongoing access to the latest features with no need to update `api-version` each month. 

**Dheera AI will send your requests to the `/v1/preview` endpoint when you set `api_version="preview"`.**

<Tabs>
<TabItem value="dheera_ai-sdk" label="Dheera AI SDK">

#### Non-streaming

```python showLineNumbers title="Azure Codex Models"
import dheera_ai

# Non-streaming response with Codex models
response = dheera_ai.responses(
    model="azure/codex-mini",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://dheera_ai8397336933.openai.azure.com",
    api_version="preview", # 👈 key difference
)

print(response)
```

#### Streaming
```python showLineNumbers title="Azure Codex Models"
import dheera_ai

# Streaming response with Codex models
response = dheera_ai.responses(
    model="azure/codex-mini",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://dheera_ai8397336933.openai.azure.com",
    api_version="preview", # 👈 key difference
)

for event in response:
    print(event)
```

</TabItem>
<TabItem value="proxy" label="OpenAI SDK with Dheera AI Proxy">

First, add this to your dheera_ai proxy config.yaml:
```yaml showLineNumbers title="Azure Codex Models"
model_list:
  - model_name: codex-mini
    dheera_ai_params:
      model: azure/codex-mini
      api_key: os.environ/AZURE_RESPONSES_OPENAI_API_KEY
      api_base: https://dheera_ai8397336933.openai.azure.com
      api_version: preview # 👈 key difference
```

Start your Dheera AI proxy:
```bash
dheera_ai --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Then use the OpenAI SDK pointed to your proxy:

#### Non-streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="codex-mini",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="codex-mini",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>
</Tabs>


## Calling via `/chat/completions`

You can also call the Azure Responses API via the `/chat/completions` endpoint.


<Tabs>
<TabItem value="dheera_ai-sdk" label="Dheera AI SDK">

```python showLineNumbers
from dheera_ai import completion
import os 

os.environ["AZURE_API_BASE"] = "https://my-endpoint-sweden-berri992.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-03-15-preview"
os.environ["AZURE_API_KEY"] = "my-api-key"

response = completion(
    model="azure/responses/my-custom-o1-pro",
    messages=[{"role": "user", "content": "Hello world"}],
)

print(response)
```
</TabItem>
<TabItem value="proxy" label="OpenAI SDK with Dheera AI Proxy">

1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: my-custom-o1-pro
    dheera_ai_params:
      model: azure/responses/my-custom-o1-pro
      api_key: os.environ/AZURE_API_KEY
      api_base: https://my-endpoint-sweden-berri992.openai.azure.com/
      api_version: 2023-03-15-preview
```

2. Start Dheera AI proxy
```bash
dheera_ai --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

```bash
curl http://localhost:4000/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DHEERA_AI_API_KEY" \
  -d '{
    "model": "my-custom-o1-pro",
    "messages": [{"role": "user", "content": "Hello world"}]
  }'
```
</TabItem>
</Tabs>