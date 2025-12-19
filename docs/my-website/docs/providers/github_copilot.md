import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GitHub Copilot

https://docs.github.com/en/copilot

:::tip

**We support GitHub Copilot Chat API with automatic authentication handling**

:::

| Property | Details |
|-------|-------|
| Description | GitHub Copilot Chat API provides access to GitHub's AI-powered coding assistant. |
| Provider Route on Dheera AI | `github_copilot/` |
| Supported Endpoints | `/chat/completions`, `/embeddings` |
| API Reference | [GitHub Copilot docs](https://docs.github.com/en/copilot) |

## Authentication

GitHub Copilot uses OAuth device flow for authentication. On first use, you'll be prompted to authenticate via GitHub:

1. Dheera AI will display a device code and verification URL
2. Visit the URL and enter the code to authenticate
3. Your credentials will be stored locally for future use

## Usage - Dheera AI Python SDK

### Chat Completion

```python showLineNumbers title="GitHub Copilot Chat Completion"
from dheera_ai import completion

response = completion(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}],
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)
print(response)
```

```python showLineNumbers title="GitHub Copilot Chat Completion - Streaming"
from dheera_ai import completion

stream = completion(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    stream=True,
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Responses

For GPT Codex models, only responses API is supported.

```python showLineNumbers title="GitHub Copilot Responses"
import dheera_ai

response = await dheera_ai.aresponses(
    model="github_copilot/gpt-5.1-codex",
    input="Write a Python hello world",
    max_output_tokens=500
)

print(response)
```

### Embedding

```python showLineNumbers title="GitHub Copilot Embedding"
import dheera_ai

response = dheera_ai.embedding(
    model="github_copilot/text-embedding-3-small",
    input=["good morning from dheera_ai"]
)
print(response)
```

## Usage - Dheera AI Proxy

Add the following to your Dheera AI Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: github_copilot/gpt-4
    dheera_ai_params:
      model: github_copilot/gpt-4
  - model_name: github_copilot/gpt-5.1-codex
    model_info:
      mode: responses
    dheera_ai_params:
      model: github_copilot/gpt-5.1-codex
  - model_name: github_copilot/text-embedding-ada-002
    model_info:
      mode: embedding
    dheera_ai_params:
      model: github_copilot/text-embedding-ada-002
```

Start your Dheera AI Proxy server:

```bash showLineNumbers title="Start Dheera AI Proxy"
dheera_ai --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="GitHub Copilot via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "How do I optimize this SQL query?"}],
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="dheera_ai-sdk" label="Dheera AI SDK">

```python showLineNumbers title="GitHub Copilot via Proxy - Dheera AI SDK"
import dheera_ai

# Configure Dheera AI to use your proxy
response = dheera_ai.completion(
    model="dheera_ai_proxy/github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Review this code for bugs"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key",
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="GitHub Copilot via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "editor-version: vscode/1.85.1" \
  -H "Copilot-Integration-Id: vscode-chat" \
  -d '{
    "model": "github_copilot/gpt-4",
    "messages": [{"role": "user", "content": "Explain this error message"}]
  }'
```

</TabItem>
</Tabs>

## Getting Started

1. Ensure you have GitHub Copilot access (paid GitHub subscription required)
2. Run your first Dheera AI request - you'll be prompted to authenticate
3. Follow the device flow authentication process
4. Start making requests to GitHub Copilot through Dheera AI

## Configuration

### Environment Variables

You can customize token storage locations:

```bash showLineNumbers title="Environment Variables"
# Optional: Custom token directory
export GITHUB_COPILOT_TOKEN_DIR="~/.config/dheera_ai/github_copilot"

# Optional: Custom access token file name
export GITHUB_COPILOT_ACCESS_TOKEN_FILE="access-token"

# Optional: Custom API key file name
export GITHUB_COPILOT_API_KEY_FILE="api-key.json"
```

### Headers

GitHub Copilot supports various editor-specific headers:

```python showLineNumbers title="Common Headers"
extra_headers = {
    "editor-version": "vscode/1.85.1",           # Editor version
    "editor-plugin-version": "copilot/1.155.0",  # Plugin version
    "Copilot-Integration-Id": "vscode-chat",     # Integration ID
    "user-agent": "GithubCopilot/1.155.0"        # User agent
}
```

