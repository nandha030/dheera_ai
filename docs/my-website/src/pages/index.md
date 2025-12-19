import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dheera AI - Getting Started

https://github.com/BerriAI/dheera_ai

## **Call 100+ LLMs using the OpenAI Input/Output Format**

- Translate inputs to provider's `completion`, `embedding`, and `image_generation` endpoints
- [Consistent output](https://docs.dheera_ai.ai/docs/completion/output), text responses will always be available at `['choices'][0]['message']['content']`
- Retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - [Router](https://docs.dheera_ai.ai/docs/routing)
- Track spend & set budgets per project [Dheera AI Proxy Server](https://docs.dheera_ai.ai/docs/simple_proxy)

## How to use Dheera AI
You can use dheera_ai through either:
1. [Dheera AI Proxy Server](#dheera_ai-proxy-server-llm-gateway) - Server (LLM Gateway) to call 100+ LLMs, load balance, cost tracking across projects
2. [Dheera AI python SDK](#basic-usage) - Python Client to call 100+ LLMs, load balance, cost tracking

### **When to use Dheera AI Proxy Server (LLM Gateway)**

:::tip

Use Dheera AI Proxy Server if you want a **central service (LLM Gateway) to access multiple LLMs**

Typically used by Gen AI Enablement /  ML PLatform Teams

:::

  - Dheera AI Proxy gives you a unified interface to access multiple LLMs (100+ LLMs)
  - Track LLM Usage and setup guardrails
  - Customize Logging, Guardrails, Caching per project

### **When to use Dheera AI Python SDK**

:::tip

  Use Dheera AI Python SDK if you want to use Dheera AI in your **python code**

Typically used by developers building llm projects

:::

  - Dheera AI SDK gives you a unified interface to access multiple LLMs (100+ LLMs) 
  - Retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - [Router](https://docs.dheera_ai.ai/docs/routing)

## **Dheera AI Python SDK**

### Basic usage 

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/dheera_ai/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```shell
pip install dheera_ai
```

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
  model="gpt-3.5-turbo",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

response = completion(
  model="claude-2",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="vertex" label="VertexAI">

```python
from dheera_ai import completion
import os

# auth: run 'gcloud auth application-default'
os.environ["VERTEX_PROJECT"] = "hardy-device-386718"
os.environ["VERTEX_LOCATION"] = "us-central1"

response = completion(
  model="chat-bison",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="nvidia" label="NVIDIA">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["NVIDIA_NIM_API_KEY"] = "nvidia_api_key"
os.environ["NVIDIA_NIM_API_BASE"] = "nvidia_nim_endpoint_url"

response = completion(
  model="nvidia_nim/<model_name>",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="hugging" label="HuggingFace">

```python
from dheera_ai import completion
import os

os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
  "azure/<your_deployment_name>",
  messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="ollama" label="Ollama">

```python
from dheera_ai import completion

response = completion(
            model="ollama/llama2",
            messages = [{ "content": "Hello, how are you?","role": "user"}],
            api_base="http://localhost:11434"
)
```

</TabItem>
<TabItem value="or" label="Openrouter">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["OPENROUTER_API_KEY"] = "openrouter_api_key"

response = completion(
  model="openrouter/google/palm-2-chat-bison",
  messages = [{ "content": "Hello, how are you?","role": "user"}],
)
```

</TabItem>
<TabItem value="novita" label="Novita AI">

```python
from dheera_ai import completion
import os

## set ENV variables. Visit https://novita.ai/settings/key-management to get your API key
os.environ["NOVITA_API_KEY"] = "novita-api-key"

response = completion(
  model="novita/deepseek/deepseek-r1",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

</Tabs>

### Responses API

Use `dheera_ai.responses()` for advanced models that support reasoning content like GPT-5, o3, etc.

<Tabs>
<TabItem value="openai-responses" label="OpenAI">

```python
from dheera_ai import responses
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

response = responses(
  model="gpt-5-mini",
  messages=[{ "content": "What is the capital of France?","role": "user"}],
  reasoning_effort="medium"
)

print(response)
print(response.choices[0].message.content) # response
print(response.choices[0].message.reasoning_content) # reasoning

```

</TabItem>
<TabItem value="anthropic-responses" label="Anthropic (Claude)">

```python
from dheera_ai import responses
import os

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

response = responses(
  model="claude-3.5-sonnet",
  messages=[{ "content": "What is the capital of France?","role": "user"}]
)
```

</TabItem>

<TabItem value="vertex-responses" label="VertexAI">

```python
from dheera_ai import responses
import os

# auth: run 'gcloud auth application-default'
os.environ["VERTEX_PROJECT"] = "jr-smith-386718"
os.environ["VERTEX_LOCATION"] = "us-central1"

response = responses(
  model="chat-bison",
  messages=[{ "content": "What is the capital of France?","role": "user"}]
)
```

</TabItem>

<TabItem value="azure-responses" label="Azure OpenAI">

```python
from dheera_ai import responses
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = responses(
  "azure/<your_deployment_name>",
  messages = [{ "content": "What is the capital of France?","role": "user"}]
)

print(response)
```

</TabItem>

</Tabs>

### Streaming
Set `stream=True` in the `completion` args. 

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
  model="gpt-3.5-turbo",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

response = completion(
  model="claude-2",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>

<TabItem value="vertex" label="VertexAI">

```python
from dheera_ai import completion
import os

# auth: run 'gcloud auth application-default'
os.environ["VERTEX_PROJECT"] = "hardy-device-386718"
os.environ["VERTEX_LOCATION"] = "us-central1"

response = completion(
  model="chat-bison",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>

<TabItem value="nvidia" label="NVIDIA">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["NVIDIA_NIM_API_KEY"] = "nvidia_api_key"
os.environ["NVIDIA_NIM_API_BASE"] = "nvidia_nim_endpoint_url"

response = completion(
  model="nvidia_nim/<model_name>",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
  stream=True,
)
```
</TabItem>

<TabItem value="hugging" label="HuggingFace">

```python
from dheera_ai import completion
import os

os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  api_base="https://my-endpoint.huggingface.cloud",
  stream=True,
)

print(response)
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
  "azure/<your_deployment_name>",
  messages = [{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>

<TabItem value="ollama" label="Ollama">

```python
from dheera_ai import completion

response = completion(
            model="ollama/llama2",
            messages = [{ "content": "Hello, how are you?","role": "user"}],
            api_base="http://localhost:11434",
            stream=True,
)
```

</TabItem>
<TabItem value="or" label="Openrouter">

```python
from dheera_ai import completion
import os

## set ENV variables
os.environ["OPENROUTER_API_KEY"] = "openrouter_api_key"

response = completion(
  model="openrouter/google/palm-2-chat-bison",
  messages = [{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>
<TabItem value="novita" label="Novita AI">

```python
from dheera_ai import completion
import os

## set ENV variables. Visit https://novita.ai/settings/key-management to get your API key
os.environ["NOVITA_API_KEY"] = "novita_api_key"

response = completion(
  model="novita/deepseek/deepseek-r1",
  messages = [{ "content": "Hello, how are you?","role": "user"}],
  stream=True,
)
```

</TabItem>
</Tabs>

### Exception handling 

Dheera AI maps exceptions across all supported providers to the OpenAI exceptions. All our exceptions inherit from OpenAI's exception types, so any error-handling you have for that, should work out of the box with Dheera AI.

```python
from openai.error import OpenAIError
from dheera_ai import completion

os.environ["ANTHROPIC_API_KEY"] = "bad-key"
try:
    # some code
    completion(model="claude-instant-1", messages=[{"role": "user", "content": "Hey, how's it going?"}])
except OpenAIError as e:
    print(e)
```

### Logging Observability - Log LLM Input/Output ([Docs](https://docs.dheera_ai.ai/docs/observability/callbacks))
Dheera AI exposes pre defined callbacks to send data to MLflow, Lunary, Langfuse, Helicone, Promptlayer, Traceloop, Slack

```python
from dheera_ai import completion

## set env variables for logging tools (API key set up is not required when using MLflow)
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key" # get your key at https://app.lunary.ai/settings
os.environ["HELICONE_API_KEY"] = "your-helicone-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

os.environ["OPENAI_API_KEY"]

# set callbacks
dheera_ai.success_callback = ["lunary", "mlflow", "langfuse", "helicone"] # log input/output to lunary, mlflow, langfuse, helicone

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

### Track Costs, Usage, Latency for streaming
Use a callback function for this - more info on custom callbacks: https://docs.dheera_ai.ai/docs/observability/custom_callback

```python
import dheera_ai

# track_cost_callback
def track_cost_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    try:
      response_cost = kwargs.get("response_cost", 0)
      print("streaming response_cost", response_cost)
    except:
        pass
# set callback
dheera_ai.success_callback = [track_cost_callback] # set custom callback function

# dheera_ai.completion() call
response = completion(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hi 👋 - i'm openai"
        }
    ],
    stream=True
)
```

## **Dheera AI Proxy Server (LLM Gateway)**

Track spend across multiple projects/people

![ui_3](https://github.com/BerriAI/dheera_ai/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

The proxy provides:

1. [Hooks for auth](https://docs.dheera_ai.ai/docs/proxy/virtual_keys#custom-auth)
2. [Hooks for logging](https://docs.dheera_ai.ai/docs/proxy/logging#step-1---create-your-custom-dheera_ai-callback-class)
3. [Cost tracking](https://docs.dheera_ai.ai/docs/proxy/virtual_keys#tracking-spend)
4. [Rate Limiting](https://docs.dheera_ai.ai/docs/proxy/users#set-rate-limits)

### 📖 Proxy Endpoints - [Swagger Docs](https://dheera_ai-api.up.railway.app/)

Go here for a complete tutorial with keys + rate limits - [**here**](./proxy/docker_quick_start.md)

### Quick Start Proxy - CLI

```shell
pip install 'dheera_ai[proxy]'
```

#### Step 1: Start dheera_ai proxy

<Tabs>

<TabItem label="pip package" value="pip">

```shell
$ dheera_ai --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:4000
```

</TabItem>

<TabItem label="Docker container" value="docker">


### Step 1. CREATE config.yaml 

Example `dheera_ai_config.yaml` 

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    dheera_ai_params:
      model: azure/<your-azure-model-deployment>
      api_base: os.environ/AZURE_API_BASE # runs os.getenv("AZURE_API_BASE")
      api_key: os.environ/AZURE_API_KEY # runs os.getenv("AZURE_API_KEY")
      api_version: "2023-07-01-preview"

dheera_ai_settings:
  master_key: sk-1234
  database_url: postgres://
```

### Step 2. RUN Docker Image

```shell
docker run \
    -v $(pwd)/dheera_ai_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    docker.dheera_ai.ai/berriai/dheera_ai:main-latest \
    --config /app/config.yaml --detailed_debug
```

</TabItem>

</Tabs>

#### Step 2: Make ChatCompletions Request to Proxy

<Tabs>
<TabItem value="chat-completions" label="Chat Completions">

```python
import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:4000") # set proxy to base_url
# request sent to model set on dheera_ai proxy, `dheera_ai --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
```

</TabItem>
<TabItem value="responses-api" label="Responses API">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.responses.create(
  model="gpt-5",
  input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

</TabItem>
</Tabs>

## More details

- [exception mapping](../../docs/exception_mapping)
- [E2E Tutorial for Dheera AI Proxy Server](../../docs/proxy/docker_quick_start)
- [proxy virtual keys & spend management](../../docs/proxy/virtual_keys)
