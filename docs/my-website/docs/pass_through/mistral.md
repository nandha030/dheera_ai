# Mistral

Pass-through endpoints for Mistral - call provider-specific endpoint, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ❌ | [Tell us if you need this](https://github.com/BerriAI/dheera_ai/issues/new) |
| Streaming | ✅ | |

Just replace `https://api.mistral.ai/v1` with `DHEERA_AI_PROXY_BASE_URL/mistral` 🚀

#### **Example Usage**

```bash
curl -L -X POST 'http://0.0.0.0:4000/mistral/v1/ocr' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "mistral-ocr-latest",
    "document": {
        "type": "image_url",
        "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
    }

}'
```

Supports **ALL** Mistral Endpoints (including streaming).

## Quick Start

Let's call the Mistral [`/chat/completions` endpoint](https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post)

1. Add MISTRAL_API_KEY to your environment 

```bash
export MISTRAL_API_KEY="sk-1234"
```

2. Start Dheera AI Proxy 

```bash
dheera_ai

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Mistral `/ocr` endpoint

```bash
curl -L -X POST 'http://0.0.0.0:4000/mistral/v1/ocr' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "mistral-ocr-latest",
    "document": {
        "type": "image_url",
        "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
    }

}'
```


## Examples

Anything after `http://0.0.0.0:4000/mistral` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://api.mistral.ai/v1`          | `http://0.0.0.0:4000/mistral` (DHEERA_AI_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `bearer $MISTRAL_API_KEY`                                 | `bearer anything` (use `bearer DHEERA_AI_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |


### **Example 1: OCR endpoint**

#### Dheera AI Proxy Call 

```bash
curl -L -X POST 'http://0.0.0.0:4000/mistral/v1/ocr' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $DHEERA_AI_API_KEY' \
-d '{
    "model": "mistral-ocr-latest",
    "document": {
        "type": "image_url",
        "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
    }
}'
```


#### Direct Mistral API Call 

```bash
curl https://api.mistral.ai/v1/ocr \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -d '{
    "model": "mistral-ocr-latest",
    "document": {
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    "include_image_base64": true
  }'
```

### **Example 2: Chat API**

#### Dheera AI Proxy Call 

```bash
curl -L -X POST 'http://0.0.0.0:4000/mistral/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $DHEERA_AI_VIRTUAL_KEY' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "mistral-large-latest",
}'
```

#### Direct Mistral API Call 

```bash
curl -L -X POST 'https://api.mistral.ai/v1/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "mistral-large-latest",
}'
```


## Advanced - Use with Virtual Keys 

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw Mistral API key, but still letting them use Mistral endpoints.

### Usage

1. Setup environment

```bash
export DATABASE_URL=""
export DHEERA_AI_MASTER_KEY=""
export MISTRAL_API_BASE=""
```

```bash
dheera_ai

# RUNNING on http://0.0.0.0:4000
```

2. Generate virtual key 

```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

Expected Response 

```bash
{
    ...
    "key": "sk-1234ewknldferwedojwojw"
}
```

3. Test it! 


```bash
curl -L -X POST 'http://0.0.0.0:4000/mistral/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234ewknldferwedojwojw' \
  --data '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "qwen2.5-7b-instruct",
}'
```