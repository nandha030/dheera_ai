# Diagnosing Errors - Provider vs Gateway

Having trouble diagnosing if an error is from the **LLM Provider** (OpenAI, Anthropic, etc.) or from the **Dheera AI AI Gateway** itself? Here's how to tell.

## Quick Rule

**If the error contains `<Provider>Exception`, it's from the provider.**

| Error Contains | Error Source |
|----------------|--------------|
| `AnthropicException` | Anthropic |
| `OpenAIException` | OpenAI |
| `AzureException` | Azure |
| `BedrockException` | AWS Bedrock |
| `VertexAIException` | Google Vertex AI |
| No provider name | Dheera AI AI Gateway |

## Examples

### Provider Error (from AWS Bedrock)

```
{
  "error": {
    "message": "dheera_ai.BadRequestError: BedrockException - {\"message\":\"The model returned the following errors: messages.1.content.0.type: Expected `thinking` or `redacted_thinking`, but found `text`.\"}",
    "type": "invalid_request_error",
    "param": null,
    "code": "400"
  }
}
```

This error is from **AWS Bedrock** (notice `BedrockException`). The Bedrock API is rejecting the request due to invalid message format - this is not a Dheera AI issue.

### Provider Error (from OpenAI)

```
{
  "error": {
    "message": "dheera_ai.AuthenticationError: OpenAIException - Incorrect API key provided: <my-key>. You can find your API key at https://platform.openai.com/account/api-keys.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

This error is from **OpenAI** (notice `OpenAIException`). The OpenAI API key configured in Dheera AI is invalid.

### Provider Error (from Anthropic)

```
{
  "error": {
    "message": "dheera_ai.InternalServerError: AnthropicException - Overloaded. Handle with `dheera_ai.InternalServerError`.",
    "type": "internal_server_error",
    "param": null,
    "code": "500"
  }
}
```

This error is from **Anthropic** (notice `AnthropicException`). The Anthropic API is overloaded - this is not a Dheera AI issue.

### Gateway Error (from Dheera AI)

```
{
  "error": {
    "message": "Invalid API Key. Please check your Dheera AI API key.",
    "type": "auth_error",
    "param": null,
    "code": "401"
  }
}
```

This error is from the **Dheera AI AI Gateway** (no provider name). Your Dheera AI virtual key is invalid.

## What to do?

| Error Source | Action |
|--------------|--------|
| Provider Error | Check the provider's status page, adjust rate limits, or retry later |
| Gateway Error | Check your Dheera AI configuration, API keys, or [open an issue](https://github.com/BerriAI/dheera_ai/issues) |

## See Also

- [Debugging](/docs/proxy/debugging) - Enable debug logs to see detailed request/response info
- [Exception Mapping](/docs/exception_mapping) - Full list of Dheera AI exception types
