import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dheera AI Proxy (LLM Gateway)


| Property | Details |
|-------|-------|
| Description | Dheera AI Proxy is an OpenAI-compatible gateway that allows you to interact with multiple LLM providers through a unified API. Simply use the `dheera_ai_proxy/` prefix before the model name to route your requests through the proxy. |
| Provider Route on Dheera AI | `dheera_ai_proxy/` (add this prefix to the model name, to route any requests to dheera_ai_proxy - e.g. `dheera_ai_proxy/your-model-name`) |
| Setup Dheera AI Gateway | [Dheera AI Gateway ↗](../simple_proxy) |
| Supported Endpoints |`/chat/completions`, `/completions`, `/embeddings`, `/audio/speech`, `/audio/transcriptions`, `/images`, `/images/edits`, `/rerank` |



## Required Variables

```python
os.environ["DHEERA_AI_PROXY_API_KEY"] = "" # "sk-1234" your dheera_ai proxy api key 
os.environ["DHEERA_AI_PROXY_API_BASE"] = "" # "http://localhost:4000" your dheera_ai proxy api base
```


## Usage (Non Streaming)
```python
import os 
import dheera_ai
from dheera_ai import completion

os.environ["DHEERA_AI_PROXY_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or dheera_ai.api_base
# os.environ["DHEERA_AI_PROXY_API_BASE"] = ""
dheera_ai.api_base = "your-openai-proxy-url"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# dheera_ai proxy call
response = completion(model="dheera_ai_proxy/your-model-name", messages)
```

## Usage - passing `api_base`, `api_key` per request

If you need to set api_base dynamically, just pass it in completions instead - completions(...,api_base="your-proxy-api-base")

```python
import os 
import dheera_ai
from dheera_ai import completion

os.environ["DHEERA_AI_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# dheera_ai proxy call
response = completion(
    model="dheera_ai_proxy/your-model-name", 
    messages=messages, 
    api_base = "your-dheera_ai-proxy-url",
    api_key = "your-dheera_ai-proxy-api-key"
)
```
## Usage - Streaming

```python
import os 
import dheera_ai
from dheera_ai import completion

os.environ["DHEERA_AI_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(
    model="dheera_ai_proxy/your-model-name", 
    messages=messages,
    api_base = "your-dheera_ai-proxy-url", 
    stream=True
)

for chunk in response:
    print(chunk)
```

## Embeddings

```python
import dheera_ai

response = dheera_ai.embedding(
    model="dheera_ai_proxy/your-embedding-model",
    input="Hello world",
    api_base="your-dheera_ai-proxy-url",
    api_key="your-dheera_ai-proxy-api-key"
)
```

## Image Generation

```python
import dheera_ai

response = dheera_ai.image_generation(
    model="dheera_ai_proxy/dall-e-3",
    prompt="A beautiful sunset over mountains",
    api_base="your-dheera_ai-proxy-url",
    api_key="your-dheera_ai-proxy-api-key"
)
```

## Image Edit

```python
import dheera_ai

with open("your-image.png", "rb") as f:
    response = dheera_ai.image_edit(
        model="dheera_ai_proxy/gpt-image-1",
        prompt="Make this image a watercolor painting",
        image=[f],
        api_base="your-dheera_ai-proxy-url",
        api_key="your-dheera_ai-proxy-api-key",
    )
```

## Audio Transcription

```python
import dheera_ai

response = dheera_ai.transcription(
    model="dheera_ai_proxy/whisper-1",
    file="your-audio-file",
    api_base="your-dheera_ai-proxy-url",
    api_key="your-dheera_ai-proxy-api-key"
)
```

## Text to Speech

```python
import dheera_ai

response = dheera_ai.speech(
    model="dheera_ai_proxy/tts-1",
    input="Hello world",
    api_base="your-dheera_ai-proxy-url",
    api_key="your-dheera_ai-proxy-api-key"
)
``` 

## Rerank

```python
import dheera_ai

import dheera_ai

response = dheera_ai.rerank(
    model="dheera_ai_proxy/rerank-english-v2.0",
    query="What is machine learning?",
    documents=[
        "Machine learning is a field of study in artificial intelligence",
        "Biology is the study of living organisms"
    ],
    api_base="your-dheera_ai-proxy-url",
    api_key="your-dheera_ai-proxy-api-key"
)
```


## Integration with Other Libraries

Dheera AI Proxy works seamlessly with Langchain, LlamaIndex, OpenAI JS, Anthropic SDK, Instructor, and more.

[Learn how to use Dheera AI proxy with these libraries →](../proxy/user_keys)

## Send all SDK requests to Dheera AI Proxy

:::info

Requires v1.72.1 or higher.

:::

Use this when calling Dheera AI Proxy from any library / codebase already using the Dheera AI SDK.

These flags will route all requests through your Dheera AI proxy, regardless of the model specified.

When enabled, requests will use `DHEERA_AI_PROXY_API_BASE` with `DHEERA_AI_PROXY_API_KEY` as the authentication.

### Option 1: Set Globally in Code

```python
# Set the flag globally for all requests
dheera_ai.use_dheera_ai_proxy = True

response = dheera_ai.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
```

### Option 2: Control via Environment Variable

```python
# Control proxy usage through environment variable
os.environ["USE_DHEERA_AI_PROXY"] = "True"

response = dheera_ai.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
```

### Option 3: Set Per Request

```python
# Enable proxy for specific requests only
response = dheera_ai.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    use_dheera_ai_proxy=True
)
```

## Sending `tags` to Dheera AI Proxy

Tags allow you to categorize and track your API requests for monitoring, debugging, and analytics purposes. You can send tags as a list of strings to the Dheera AI Proxy using the `extra_body` parameter.

### Usage

Send tags by including them in the `extra_body` parameter of your completion request:

```python showLineNumbers title="Usage"
import dheera_ai

response = dheera_ai.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_base="http://localhost:4000",
    api_key="sk-1234",
    extra_body={"tags": ["user:ishaan", "department:engineering", "priority:high"]}
)
```

### Async Usage

```python showLineNumbers title="Async Usage"
import dheera_ai

response = await dheera_ai.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_base="http://localhost:4000", 
    api_key="sk-1234",
    extra_body={"tags": ["user:ishaan", "department:engineering"]}
)
```

