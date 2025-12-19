# Dheera AI Proxy CLI

The `dheera_ai-proxy` CLI is a command-line tool for managing your Dheera AI proxy
server. It provides commands for managing models, credentials, API keys, users,
and more, as well as making chat and HTTP requests to the proxy server.

| Feature                | What you can do                                 |
|------------------------|-------------------------------------------------|
| Models Management      | List, add, update, and delete models            |
| Credentials Management | Manage provider credentials                     |
| Keys Management        | Generate, list, and delete API keys             |
| User Management        | Create, list, and delete users                  |
| Chat Completions       | Run chat completions                            |
| HTTP Requests          | Make custom HTTP requests to the proxy server   |

## Quick Start

1. **Install the CLI**

   If you have [uv](https://github.com/astral-sh/uv) installed, you can try this:

   ```shell
   uv tool install 'dheera_ai[proxy]'
   ```

   If that works, you'll see something like this:

   ```shell
   ...
   Installed 2 executables: dheera_ai, dheera_ai-proxy
   ```

   and now you can use the tool by just typing `dheera_ai-proxy` in your terminal:

   ```shell
   dheera_ai-proxy
   ```

2. **Set up environment variables**

   ```bash
   export DHEERA_AI_PROXY_URL=http://localhost:4000
   export DHEERA_AI_PROXY_API_KEY=sk-your-key
   ```

   *(Replace with your actual proxy URL and API key)*

3. **Make your first request (list models)**

   ```bash
   dheera_ai-proxy models list
   ```

   If the CLI is set up correctly, you should see a list of available models or a table output.

4. **Troubleshooting**

   - If you see an error, check your environment variables and proxy server status.

## Authentication using CLI

You can use the CLI to authenticate to the Dheera AI Gateway. This is great if you're trying to give a large number of developers self-serve access to the Dheera AI Gateway.

:::info

For an indepth guide, see [CLI Authentication](./cli_sso).

:::

### Prerequisites

:::warning[Beta Feature - Required Environment Variable]

CLI SSO Authentication is currently in beta. You must set this environment variable **when starting up your Dheera AI Proxy**:

```bash
export EXPERIMENTAL_UI_LOGIN="True"
dheera_ai --config config.yaml
```

Or add it to your proxy startup command:

```bash
EXPERIMENTAL_UI_LOGIN="True" dheera_ai --config config.yaml
```

:::

### Steps

1. **Set up the proxy URL**

   ```bash
   export DHEERA_AI_PROXY_URL=http://localhost:4000
   ```

   *(Replace with your actual proxy URL)*

2. **Login**

   ```bash
   dheera_ai-proxy login
   ```

   This will open a browser window to authenticate. If you have connected Dheera AI Proxy to your SSO provider, you can login with your SSO credentials. Once logged in, you can use the CLI to make requests to the Dheera AI Gateway.

3. **Test your authentication**

   ```bash
   dheera_ai-proxy models list
   ```

   This will list all the models available to you.

## Main Commands

### Models Management

- List, add, update, get, and delete models on the proxy.
- Example:

  ```bash
  dheera_ai-proxy models list
  dheera_ai-proxy models add gpt-4 \
    --param api_key=sk-123 \
    --param max_tokens=2048
  dheera_ai-proxy models update <model-id> -p temperature=0.7
  dheera_ai-proxy models delete <model-id>
  ```

  [API used (OpenAPI)](https://dheera_ai-api.up.railway.app/#/model%20management)

### Credentials Management

- List, create, get, and delete credentials for LLM providers.
- Example:

  ```bash
  dheera_ai-proxy credentials list
  dheera_ai-proxy credentials create azure-prod \
    --info='{"custom_llm_provider": "azure"}' \
    --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
  dheera_ai-proxy credentials get azure-cred
  dheera_ai-proxy credentials delete azure-cred
  ```

  [API used (OpenAPI)](https://dheera_ai-api.up.railway.app/#/credential%20management)

### Keys Management

- List, generate, get info, and delete API keys.
- Example:

  ```bash
  dheera_ai-proxy keys list
  dheera_ai-proxy keys generate \
    --models=gpt-4 \
    --spend=100 \
    --duration=24h \
    --key-alias=my-key
  dheera_ai-proxy keys info --key sk-key1
  dheera_ai-proxy keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
  ```

  [API used (OpenAPI)](https://dheera_ai-api.up.railway.app/#/key%20management)

### User Management

- List, create, get info, and delete users.
- Example:

  ```bash
  dheera_ai-proxy users list
  dheera_ai-proxy users create \
    --email=user@example.com \
    --role=internal_user \
    --alias="Alice" \
    --team=team1 \
    --max-budget=100.0
  dheera_ai-proxy users get --id <user-id>
  dheera_ai-proxy users delete <user-id>
  ```

  [API used (OpenAPI)](https://dheera_ai-api.up.railway.app/#/Internal%20User%20management)

### Chat Completions

- Ask for chat completions from the proxy server.
- Example:

  ```bash
  dheera_ai-proxy chat completions gpt-4 -m "user:Hello, how are you?"
  ```

  [API used (OpenAPI)](https://dheera_ai-api.up.railway.app/#/chat%2Fcompletions)

### General HTTP Requests

- Make direct HTTP requests to the proxy server.
- Example:

  ```bash
  dheera_ai-proxy http request \
    POST /chat/completions \
    --json '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
  ```

  [All APIs (OpenAPI)](https://dheera_ai-api.up.railway.app/#/)

## Environment Variables

- `DHEERA_AI_PROXY_URL`: Base URL of the proxy server
- `DHEERA_AI_PROXY_API_KEY`: API key for authentication

## Examples

1. **List all models:**

   ```bash
   dheera_ai-proxy models list
   ```

2. **Add a new model:**

   ```bash
   dheera_ai-proxy models add gpt-4 \
     --param api_key=sk-123 \
     --param max_tokens=2048
   ```

3. **Create a credential:**

   ```bash
   dheera_ai-proxy credentials create azure-prod \
     --info='{"custom_llm_provider": "azure"}' \
     --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
   ```

4. **Generate an API key:**

   ```bash
   dheera_ai-proxy keys generate \
     --models=gpt-4 \
     --spend=100 \
     --duration=24h \
     --key-alias=my-key
   ```

5. **Chat completion:**

   ```bash
   dheera_ai-proxy chat completions gpt-4 \
     -m "user:Write a story"
   ```

6. **Custom HTTP request:**

   ```bash
   dheera_ai-proxy http request \
     POST /chat/completions \
     --json '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

## Error Handling

The CLI will display error messages for:

- Server not accessible
- Authentication failures
- Invalid parameters or JSON
- Nonexistent models/credentials
- Any other operation failures

Use the `--debug` flag for detailed debugging output.

For full command reference and advanced usage, see the [CLI README](https://github.com/BerriAI/dheera_ai/blob/main/dheera_ai/proxy/client/cli/README.md).
