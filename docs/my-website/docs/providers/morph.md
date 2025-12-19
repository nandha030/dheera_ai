# Morph

Dheera AI supports all models on [Morph](https://morphllm.com)

## Overview

Morph provides specialized AI models designed for agentic workflows, particularly excelling at precise code editing and manipulation. Their "Apply" models enable targeted code changes without full file rewrites, making them ideal for AI agents that need to make intelligent, context-aware code modifications.

## API Key
```python
import os 
os.environ["MORPH_API_KEY"] = "your-api-key"
```

## Sample Usage

```python
from dheera_ai import completion

# set env variable 
os.environ["MORPH_API_KEY"] = "your-api-key"

messages = [
    {"role": "user", "content": "Write a Python function to calculate factorial"}
]

## Morph v3 Fast - Optimized for speed
response = completion(
    model="morph/morph-v3-fast",
    messages=messages,
)
print(response)

## Morph v3 Large - Most capable model
response = completion(
    model="morph/morph-v3-large", 
    messages=messages,
)
print(response)
```

## Sample Usage - Streaming
```python
from dheera_ai import completion

# set env variable
os.environ["MORPH_API_KEY"] = "your-api-key"

messages = [
    {"role": "user", "content": "Write a Python function to calculate factorial"}
]

## Morph v3 Fast with streaming
response = completion(
    model="morph/morph-v3-fast",
    messages=messages,
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Supported Models

| Model Name               | Function Call                              | Description | Context Window |
|--------------------------|--------------------------------------------|-----------------------|----------------|
| morph-v3-fast            | `completion('morph/morph-v3-fast', messages)` | Fastest model, optimized for quick responses | 16k tokens |
| morph-v3-large           | `completion('morph/morph-v3-large', messages)` | Most capable model for complex tasks | 16k tokens |

## Usage - Dheera AI Proxy Server

Here's how to use Morph with the Dheera AI Proxy Server:

1. Save API key in your environment
```bash
export MORPH_API_KEY="your-api-key"
```

2. Add model to config.yaml
```yaml
model_list:
  - model_name: morph-v3-fast
    dheera_ai_params:
      model: morph/morph-v3-fast
      
  - model_name: morph-v3-large
    dheera_ai_params:
      model: morph/morph-v3-large
```

3. Start the proxy server
```bash
dheera_ai --config config.yaml
```

## Advanced Usage

### Setting API Base
```python
import dheera_ai 

# set custom api base
response = completion(
    model="morph/morph-v3-large",
    messages=[{"role": "user", "content": "Hello, world!"}],
    api_base="https://api.morphllm.com/v1"
)
print(response)
```

### Setting API Key
```python 
import dheera_ai 

# set api key via completion
response = completion(
    model="morph/morph-v3-large",
    messages=[{"role": "user", "content": "Hello, world!"}],
    api_key="your-api-key"
)
print(response)
```