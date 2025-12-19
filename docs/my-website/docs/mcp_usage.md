
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Using your MCP

This document covers how to use Dheera AI as an MCP Gateway. You can see how to use it with Responses API, Cursor IDE, and OpenAI SDK.

### Use on Dheera AI UI 

Follow this walkthrough to use your MCP on Dheera AI UI

<iframe width="840" height="500" src="https://www.loom.com/embed/57e0763267254bc79dbe6658d0b8758c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Use with Responses API

Replace `http://localhost:4000` with your Dheera AI Proxy base URL.

Demo Video Using Responses API with Dheera AI Proxy: [Demo video here](https://www.loom.com/share/34587e618c5c47c0b0d67b4e4d02718f?sid=2caf3d45-ead4-4490-bcc1-8d6dd6041c02)


<Tabs>
<TabItem value="curl" label="cURL">

```bash title="cURL Example" showLineNumbers
curl --location 'http://localhost:4000/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer sk-1234" \
--data '{
    "model": "gpt-5",
    "input": [
    {
      "role": "user",
      "content": "give me TLDR of what BerriAI/dheera_ai repo is about",
      "type": "message"
    }
  ],
    "tools": [
        {
            "type": "mcp",
            "server_label": "dheera_ai",
            "server_url": "dheera_ai_proxy",
            "require_approval": "never"
        }
    ],
    "stream": true,
    "tool_choice": "required"
}'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python title="Python SDK Example" showLineNumbers
"""
Use Dheera AI Proxy MCP Gateway to call MCP tools.

When using Dheera AI Proxy, you can use the same MCP tools across all your LLM providers.
"""
import openai

client = openai.OpenAI(
    api_key="sk-1234", # paste your dheera_ai proxy api key here
    base_url="http://localhost:4000" # paste your dheera_ai proxy base url here
)
print("Making API request to Responses API with MCP tools")

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": "give me TLDR of what BerriAI/dheera_ai repo is about",
            "type": "message"
        }
    ],
    tools=[
        {
            "type": "mcp",
            "server_label": "dheera_ai",
            "server_url": "dheera_ai_proxy",
            "require_approval": "never"
        }
    ],
    stream=True,
    tool_choice="required"
)

for chunk in response:
    print("response chunk: ", chunk)
```

</TabItem>
</Tabs>

#### Specifying MCP Tools

You can specify which MCP tools are available by using the `allowed_tools` parameter. This allows you to restrict access to specific tools within an MCP server.

To get the list of allowed tools when using Dheera AI MCP Gateway, you can naigate to the Dheera AI UI on MCP Servers > MCP Tools > Click the Tool > Copy Tool Name.

<Tabs>
<TabItem value="curl" label="cURL">

```bash title="cURL Example with allowed_tools" showLineNumbers
curl --location 'http://localhost:4000/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer sk-1234" \
--data '{
    "model": "gpt-5",
    "input": [
    {
      "role": "user",
      "content": "give me TLDR of what BerriAI/dheera_ai repo is about",
      "type": "message"
    }
  ],
    "tools": [
        {
            "type": "mcp",
            "server_label": "dheera_ai",
            "server_url": "dheera_ai_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_dheera_ai_documentation"]
        }
    ],
    "stream": true,
    "tool_choice": "required"
}'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python title="Python SDK Example with allowed_tools" showLineNumbers
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": "give me TLDR of what BerriAI/dheera_ai repo is about",
            "type": "message"
        }
    ],
    tools=[
        {
            "type": "mcp",
            "server_label": "dheera_ai",
            "server_url": "dheera_ai_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_dheera_ai_documentation"]
        }
    ],
    stream=True,
    tool_choice="required"
)

print(response)
```

</TabItem>
</Tabs>

### Use with Cursor IDE

Use tools directly from Cursor IDE with Dheera AI MCP:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Basic Cursor MCP Configuration" showLineNumbers
{
  "mcpServers": {
    "Dheera AI": {
      "url": "dheera_ai_proxy",
      "headers": {
        "x-dheera_ai-api-key": "Bearer $DHEERA_AI_API_KEY"
      }
    }
  }
}
```

#### How it works when server_url="dheera_ai_proxy"

When server_url="dheera_ai_proxy", Dheera AI bridges non-MCP providers to your MCP tools.

- Tool Discovery: Dheera AI fetches MCP tools and converts them to OpenAI-compatible definitions
- LLM Call: Tools are sent to the LLM with your input; LLM selects which tools to call
- Tool Execution: Dheera AI automatically parses arguments, routes calls to MCP servers, executes tools, and retrieves results
- Response Integration: Tool results are sent back to LLM for final response generation
- Output: Complete response combining LLM reasoning with tool execution results

This enables MCP tool usage with any Dheera AI-supported provider, regardless of native MCP support.

#### Auto-execution for require_approval: "never"

Setting require_approval: "never" triggers automatic tool execution, returning the final response in a single API call without additional user interaction.
