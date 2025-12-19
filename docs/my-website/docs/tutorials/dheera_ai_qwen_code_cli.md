# Qwen Code CLI

This tutorial shows you how to integrate the Qwen Code CLI with Dheera AI Proxy, allowing you to route requests through Dheera AI's unified interface.


:::info 

This integration is supported from Dheera AI v1.73.3-nightly and above.

:::

<br />

<iframe width="840" height="500" src="https://www.loom.com/embed/d7059b059c0f425fb0b8839418adffd6" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Benefits of using qwen-code with Dheera AI

When you use qwen-code with Dheera AI you get the following benefits:

**Developer Benefits:**
- Universal Model Access: Use any Dheera AI supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the qwen-code interface.
- Higher Rate Limits & Reliability: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails.

**Proxy Admin Benefits:**
- Centralized Management: Control access to all models through a single Dheera AI proxy instance without giving your developers API Keys to each provider.
- Budget Controls: Set spending limits and track costs across all qwen-code usage.



## Prerequisites

Before you begin, ensure you have:
- Node.js and npm installed on your system
- A running Dheera AI Proxy instance
- A valid Dheera AI Proxy API key
- Git installed for cloning the repository

## Quick Start Guide

### Step 1: Install Qwen Code CLI

Clone the Qwen Code CLI repository and navigate to the project directory:

```bash
npm install -g @qwen-code/qwen-code
```

### Step 2: Configure Qwen Code CLI for Dheera AI Proxy

Configure the Qwen Code CLI to point to your Dheera AI Proxy instance by setting the required environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:4000"
export OPENAI_API_KEY=sk-1234567890
export OPENAI_MODEL="your-configured-model"
```

**Note:** Replace the values with your actual Dheera AI Proxy configuration:
- `OPENAI_BASE_URL`: The URL where your Dheera AI Proxy is running
- `OPENAI_API_KEY`: Your Dheera AI Proxy API key
- `OPENAI_MODEL`: The model you want to use (configured in your Dheera AI proxy)

### Step 3: Build and Start Qwen Code CLI

Build the project and start the CLI:

```bash
qwen
```

### Step 4: Test the Integration

Once the CLI is running, you can send test requests. These requests will be automatically routed through Dheera AI Proxy to the configured Qwen model.

The CLI will now use Dheera AI Proxy as the backend, giving you access to Dheera AI's features like:
- Request/response logging
- Rate limiting
- Cost tracking
- Model routing and fallbacks


## Advanced

### Use Anthropic, OpenAI, Bedrock, etc. models on qwen-code

In order to use non-qwen models on qwen-code, you need to set a `model_group_alias` in the Dheera AI Proxy config. This tells Dheera AI that requests with model = `qwen-code` should be routed to your desired model from any provider.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="anthropic" label="Anthropic">

Route `qwen-code` requests to Claude Sonnet:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: claude-sonnet-4-20250514
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  model_group_alias: {"qwen-code": "claude-sonnet-4-20250514"}
```

</TabItem>
<TabItem value="openai" label="OpenAI">

Route `qwen-code` requests to GPT-4o:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: gpt-4o-model
    dheera_ai_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  model_group_alias: {"qwen-code": "gpt-4o-model"}
```

</TabItem>
<TabItem value="bedrock" label="Bedrock">

Route `qwen-code` requests to Claude on Bedrock:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: bedrock-claude
    dheera_ai_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

router_settings:
  model_group_alias: {"qwen-code": "bedrock-claude"}
```

</TabItem>
<TabItem value="multi-provider" label="Multi-Provider Load Balancing">

All deployments with model_name=`anthropic-claude` will be load balanced. In this example we load balance between Anthropic and Bedrock.

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: anthropic-claude
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY  
  - model_name: anthropic-claude
    dheera_ai_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

router_settings:
  model_group_alias: {"qwen-code": "anthropic-claude"}
```

</TabItem>
</Tabs>

With this configuration, when you use `qwen-code` in the CLI, Dheera AI will automatically route your requests to the configured provider(s) with load balancing and fallbacks.





## Troubleshooting

If you encounter issues:

1. **Connection errors**: Verify that your Dheera AI Proxy is running and accessible at the configured `OPENAI_BASE_URL`
2. **Authentication errors**: Ensure your `OPENAI_API_KEY` is valid and has the necessary permissions
3. **Build failures**: Make sure all dependencies are installed with `npm install`
