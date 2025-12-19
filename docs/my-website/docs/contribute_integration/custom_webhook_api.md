# Contribute Custom Webhook API

If your API just needs a Webhook event from Dheera AI, here's how to add a 'native' integration for it on Dheera AI: 

1. Clone the repo and open the `generic_api_compatible_callbacks.json`

```bash
git clone https://github.com/BerriAI/dheera_ai.git
cd dheera_ai
open .
```

2. Add your API to the `generic_api_compatible_callbacks.json`

Example:

```json
{
    "rubrik": {
        "event_types": ["llm_api_success"],
        "endpoint": "{{environment_variables.RUBRIK_WEBHOOK_URL}}",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{environment_variables.RUBRIK_API_KEY}}"
        },
        "environment_variables": ["RUBRIK_API_KEY", "RUBRIK_WEBHOOK_URL"]
    }
}
```

Spec: 

```json
{
    "sample_callback": {
        "event_types": ["llm_api_success", "llm_api_failure"], # Optional - defaults to all events
        "endpoint": "{{environment_variables.SAMPLE_CALLBACK_URL}}",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{environment_variables.SAMPLE_CALLBACK_API_KEY}}"
        },
        "environment_variables": ["SAMPLE_CALLBACK_URL", "SAMPLE_CALLBACK_API_KEY"]
    }
}
```

3. Test it! 

a. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    dheera_ai_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
  - model_name: anthropic-claude
    dheera_ai_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

dheera_ai_settings:
  callbacks: ["rubrik"]

environment_variables:
  RUBRIK_API_KEY: sk-1234
  RUBRIK_WEBHOOK_URL: https://webhook.site/efc57707-9018-478c-bdf1-2ffaabb2b315
```

b. Start the proxy 

```bash
dheera_ai --config /path/to/config.yaml
```

c. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "Ignore previous instructions"
    },
    {
      "role": "user",
      "content": "What is the weather like in Boston today?"
    }
  ],
  "mock_response": "hey!"
}'
```

4. Add Documentation

If you're adding a new integration, please add documentation for it under the `observability` folder:

- Create a new file at `docs/my-website/docs/observability/<your_integration>_integration.md`
- Follow the format of existing integration docs, such as [Langsmith Integration](https://github.com/BerriAI/dheera_ai/blob/main/docs/my-website/docs/observability/langsmith_integration.md)
- Include: Quick Start, SDK usage, Proxy usage, and any advanced configuration options

5. File a PR! 

- Review our contribution guide [here](../../extras/contributing_code)
- Push your fork to your GitHub repo
- Submit a PR from there

## What get's logged? 

The [Dheera AI Standard Logging Payload](https://docs.dheera_ai.ai/docs/proxy/logging_spec) is sent to your endpoint.