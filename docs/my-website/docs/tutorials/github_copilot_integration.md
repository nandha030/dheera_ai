---
sidebar_label: "GitHub Copilot"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GitHub Copilot

This tutorial shows you how to integrate GitHub Copilot with Dheera AI Proxy, allowing you to route requests through Dheera AI's unified interface.

:::info 

This tutorial is based on [Sergio Pino's excellent guide](https://dev.to/spino327/calling-github-copilot-models-from-openhands-using-dheera_ai-proxy-1hl4) for calling GitHub Copilot models through Dheera AI Proxy. This integration allows you to use any Dheera AI supported model through GitHub Copilot's interface.

:::

## Benefits of using GitHub Copilot with Dheera AI

When you use GitHub Copilot with Dheera AI you get the following benefits:

**Developer Benefits:**
- Universal Model Access: Use any Dheera AI supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the GitHub Copilot interface.
- Higher Rate Limits & Reliability: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails.

**Proxy Admin Benefits:**
- Centralized Management: Control access to all models through a single Dheera AI proxy instance without giving your developers API Keys to each provider.
- Budget Controls: Set spending limits and track costs across all GitHub Copilot usage.

## Prerequisites

Before you begin, ensure you have:
- GitHub Copilot subscription (Individual, Business, or Enterprise)
- A running Dheera AI Proxy instance
- A valid Dheera AI Proxy API key
- VS Code or compatible IDE with GitHub Copilot extension

## Quick Start Guide

### Step 1: Install Dheera AI

Install Dheera AI with proxy support:

```bash
pip install dheera_ai[proxy]
```

### Step 2: Configure Dheera AI Proxy

Create a `config.yaml` file with your model configurations:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    dheera_ai_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-5-sonnet
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: sk-1234567890 # Change this to a secure key
```

### Step 3: Start Dheera AI Proxy

Start the proxy server:

```bash
dheera_ai --config config.yaml --port 4000
```

### Step 4: Configure GitHub Copilot

Configure GitHub Copilot to use your Dheera AI proxy. Add the following to your VS Code `settings.json`:

```json
{
  "github.copilot.advanced": {
    "debug.overrideProxyUrl": "http://localhost:4000",
    "debug.testOverrideProxyUrl": "http://localhost:4000"
  }
}
```

### Step 5: Test the Integration

Restart VS Code and test GitHub Copilot. Your requests will now be routed through Dheera AI Proxy, giving you access to Dheera AI's features like:
- Request/response logging
- Rate limiting
- Cost tracking
- Model routing and fallbacks

## Advanced

### Use Anthropic, OpenAI, Bedrock, etc. models with GitHub Copilot

You can route GitHub Copilot requests to any provider by configuring different models in your Dheera AI Proxy config:

<Tabs>
<TabItem value="anthropic" label="Anthropic">

Route requests to Claude Sonnet:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-3-5-sonnet
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: sk-1234567890
```

</TabItem>
<TabItem value="openai" label="OpenAI">

Route requests to GPT-4o:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    dheera_ai_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: sk-1234567890
```

</TabItem>
<TabItem value="bedrock" label="Bedrock">

Route requests to Claude on Bedrock:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: bedrock-claude
    dheera_ai_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

general_settings:
  master_key: sk-1234567890
```

</TabItem>
<TabItem value="multi-provider" label="Multi-Provider Load Balancing">

All deployments with the same model_name will be load balanced. In this example we load balance between OpenAI and Anthropic:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    dheera_ai_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-4o  # Same model name for load balancing
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: simple-shuffle

general_settings:
  master_key: sk-1234567890
```

</TabItem>
</Tabs>

With this configuration, GitHub Copilot will automatically route requests through Dheera AI to your configured provider(s) with load balancing and fallbacks.

## Troubleshooting

If you encounter issues:

1. **GitHub Copilot not using proxy**: Verify the proxy URL is correctly configured in VS Code settings and that Dheera AI proxy is running
2. **Authentication errors**: Ensure your master key is valid and API keys for providers are correctly set
3. **Connection errors**: Check that your Dheera AI Proxy is accessible at `http://localhost:4000`

## Credits

This tutorial is based on the work by [Sergio Pino](https://dev.to/spino327) from his original article: [Calling GitHub Copilot models from OpenHands using Dheera AI Proxy](https://dev.to/spino327/calling-github-copilot-models-from-openhands-using-dheera_ai-proxy-1hl4). Thank you for the foundational work! 