import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /embeddings

## Quick Start
```python
from dheera_ai import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding(model='text-embedding-ada-002', input=["good morning from dheera_ai"])
```

## Async Usage - `aembedding()`

Dheera AI provides an asynchronous version of the `embedding` function called `aembedding`:

```python
from dheera_ai import aembedding
import asyncio

async def get_embedding():
    response = await aembedding(
        model='text-embedding-ada-002',
        input=["good morning from dheera_ai"]
    )
    return response

response = asyncio.run(get_embedding())
print(response)
```

## Proxy Usage 

**NOTE**
For `vertex_ai`,
```bash
export GOOGLE_APPLICATION_CREDENTIALS="absolute/path/to/service_account.json"
```

### Add model to config 

```yaml
model_list:
- model_name: textembedding-gecko
  dheera_ai_params:
    model: vertex_ai/textembedding-gecko

general_settings:
  master_key: sk-1234
```

### Start proxy 

```bash
dheera_ai --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:4000
```

### Test 

<Tabs>
<TabItem value="curl" label="Curl">

```bash
curl --location 'http://0.0.0.0:4000/embeddings' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"input": ["Academia.edu uses"], "model": "textembedding-gecko", "encoding_format": "base64"}'
```

</TabItem>
<TabItem value="openai" label="OpenAI (python)">

```python
from openai import OpenAI
client = OpenAI(
  api_key="sk-1234",
  base_url="http://0.0.0.0:4000"
)

client.embeddings.create(
  model="textembedding-gecko",
  input="The food was delicious and the waiter...",
  encoding_format="float"
)
```
</TabItem>
<TabItem value="langchain" label="Langchain Embeddings">

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="textembedding-gecko", openai_api_base="http://0.0.0.0:4000", openai_api_key="sk-1234")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"VERTEX AI EMBEDDINGS")
print(query_result[:5])
```
</TabItem>
</Tabs>


## Image Embeddings

For models that support image embeddings, you can pass in a base64 encoded image string to the `input` param.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from dheera_ai import embedding
import os

# set your api key
os.environ["COHERE_API_KEY"] = ""

response = embedding(model="cohere/embed-english-v3.0", input=["<base64 encoded image>"])
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml 

```yaml
model_list:
  - model_name: cohere-embed
    dheera_ai_params:
      model: cohere/embed-english-v3.0
      api_key: os.environ/COHERE_API_KEY
```


2. Start proxy

```bash
dheera_ai --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/v1/embeddings' \
-H 'Authorization: Bearer sk-54d77cd67b9febbb' \
-H 'Content-Type: application/json' \
-d '{
  "model": "cohere/embed-english-v3.0",
  "input": ["<base64 encoded image>"]
}'
```
</TabItem>
</Tabs>

## Input Params for `dheera_ai.embedding()`


:::info

Any non-openai params, will be treated as provider-specific params, and sent in the request body as kwargs to the provider.

[**See Reserved Params**](https://github.com/BerriAI/dheera_ai/blob/2f5f85cb52f36448d1f8bbfbd3b8af8167d0c4c8/dheera_ai/main.py#L3130)

[**See Example**](#example)
:::

### Required Fields

- `model`: *string* - ID of the model to use. `model='text-embedding-ada-002'`

- `input`: *string or array* - Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for text-embedding-ada-002), cannot be an empty string, and any array must be 2048 dimensions or less. 
```python
input=["good morning from dheera_ai"]
```

### Optional Dheera AI Fields

- `user`: *string (optional)* A unique identifier representing your end-user, 

- `dimensions`: *integer (Optional)* The number of dimensions the resulting output embeddings should have. Only supported in OpenAI/Azure text-embedding-3 and later models.

- `encoding_format`: *string (Optional)* The format to return the embeddings in. Can be either `"float"` or `"base64"`. Defaults to `encoding_format="float"`

- `timeout`: *integer (Optional)* - The maximum time, in seconds, to wait for the API to respond. Defaults to 600 seconds (10 minutes).

- `api_base`: *string (optional)* - The api endpoint you want to call the model with

- `api_version`: *string (optional)* - (Azure-specific) the api version for the call

- `api_key`: *string (optional)* - The API key to authenticate and authorize requests. If not provided, the default API key is used.

- `api_type`: *string (optional)* - The type of API to use.

### Output from `dheera_ai.embedding()`

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.0022326677571982145,
        0.010749882087111473,
        ...
        ...
        ...
   
      ]
    }
  ],
  "model": "text-embedding-ada-002-v2",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## OpenAI Embedding Models

### Usage
```python
from dheera_ai import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding(
    model="text-embedding-3-small",
    input=["good morning from dheera_ai", "this is another item"],
    metadata={"anything": "good day"},
    dimensions=5 # Only supported in text-embedding-3 and later models.
)
```

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-3-small | `embedding('text-embedding-3-small', input)` | `os.environ['OPENAI_API_KEY']`       |
| text-embedding-3-large | `embedding('text-embedding-3-large', input)` | `os.environ['OPENAI_API_KEY']`       |
| text-embedding-ada-002 | `embedding('text-embedding-ada-002', input)` | `os.environ['OPENAI_API_KEY']`       |

## OpenAI Compatible Embedding Models
Use this for calling `/embedding` endpoints on OpenAI Compatible Servers, example https://github.com/xorbitsai/inference

**Note add `openai/` prefix to model so dheera_ai knows to route to OpenAI**

### Usage
```python
from dheera_ai import embedding
response = embedding(
  model = "openai/<your-llm-name>",     # add `openai/` prefix to model so dheera_ai knows to route to OpenAI
  api_base="http://0.0.0.0:4000/"       # set API Base of your Custom OpenAI Endpoint
  input=["good morning from dheera_ai"]
)
```

## Bedrock Embedding

### API keys
This can be set as env variables or passed as **params to dheera_ai.embedding()**
```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""  # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = "" # us-east-1, us-east-2, us-west-1, us-west-2
```

### Usage
```python
from dheera_ai import embedding
response = embedding(
    model="amazon.titan-embed-text-v1",
    input=["good morning from dheera_ai"],
)
print(response)
```

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| Amazon Nova Multimodal Embeddings | `embedding(model="bedrock/amazon.nova-2-multimodal-embeddings-v1:0", input=input)` | [Nova Docs](../providers/bedrock_embedding#amazon-nova-multimodal-embeddings) |
| Amazon Nova (Async) | `embedding(model="bedrock/async_invoke/amazon.nova-2-multimodal-embeddings-v1:0", input=input, input_type="text", output_s3_uri="s3://bucket/")` | [Nova Async Docs](../providers/bedrock_embedding#asynchronous-embeddings-with-segmentation) |
| Titan Embeddings - G1 | `embedding(model="amazon.titan-embed-text-v1", input=input)` |
| Cohere Embeddings - English | `embedding(model="cohere.embed-english-v3", input=input)` |
| Cohere Embeddings - Multilingual | `embedding(model="cohere.embed-multilingual-v3", input=input)` |
| TwelveLabs Marengo (Async) | `embedding(model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0", input=input, input_type="text")` | [Async Invoke Docs](../providers/bedrock_embedding#async-invoke-embedding) |

## TwelveLabs Bedrock Embedding Models

TwelveLabs Marengo models support multimodal embeddings (text, image, video, audio) and require the `input_type` parameter to specify the input format.

### Usage

```python
from dheera_ai import embedding
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = "us-east-1"

# Text embedding
response = embedding(
    model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["Hello world from Dheera AI!"],
    input_type="text"  # Required parameter
)

# Image embedding (base64)
response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."],
    input_type="image",  # Required parameter
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)

# Video embedding (S3 URL)
response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["s3://your-bucket/video.mp4"],
    input_type="video",  # Required parameter
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)
```

### Required Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `input_type` | Type of input content | `"text"`, `"image"`, `"video"`, `"audio"` |

### Supported Models

| Model Name | Function Call | Notes |
|------------|---------------|-------|
| TwelveLabs Marengo 2.7 (Sync) | `embedding(model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0", input=input, input_type="text")` | Text embeddings only |
| TwelveLabs Marengo 2.7 (Async) | `embedding(model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0", input=input, input_type="text/image/video/audio")` | All input types, requires `output_s3_uri` |

## Cohere Embedding Models
https://docs.cohere.com/reference/embed

### Usage
```python
from dheera_ai import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from dheera_ai", "this is another item"], 
    input_type="search_document" # optional param for v3 llms
)
```
| Model Name               | Function Call                                                |
|--------------------------|--------------------------------------------------------------|
| embed-english-v3.0       | `embedding(model="embed-english-v3.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-english-light-v3.0 | `embedding(model="embed-english-light-v3.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-multilingual-v3.0  | `embedding(model="embed-multilingual-v3.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-multilingual-light-v3.0 | `embedding(model="embed-multilingual-light-v3.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-english-v2.0       | `embedding(model="embed-english-v2.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-english-light-v2.0 | `embedding(model="embed-english-light-v2.0", input=["good morning from dheera_ai", "this is another item"])` |
| embed-multilingual-v2.0  | `embedding(model="embed-multilingual-v2.0", input=["good morning from dheera_ai", "this is another item"])` |

## NVIDIA NIM Embedding Models

### API keys
This can be set as env variables or passed as **params to dheera_ai.embedding()**
```python
import os
os.environ["NVIDIA_NIM_API_KEY"] = ""  # api key
os.environ["NVIDIA_NIM_API_BASE"] = "" # nim endpoint url
```

### Usage
```python
from dheera_ai import embedding
import os
os.environ['NVIDIA_NIM_API_KEY'] = ""
response = embedding(
    model='nvidia_nim/<model_name>', 
    input=["good morning from dheera_ai"],
    input_type="query"
)
```
## `input_type` Parameter for Embedding Models

Certain embedding models, such as `nvidia/embed-qa-4` and the E5 family, operate in **dual modes**—one for **indexing documents (passages)** and another for **querying**. To maintain high retrieval accuracy, it's essential to specify how the input text is being used by setting the `input_type` parameter correctly.

### Usage

Set the `input_type` parameter to one of the following values:

- `"passage"` – for embedding content during **indexing** (e.g., documents).
- `"query"` – for embedding content during **retrieval** (e.g., user queries).

> **Warning:** Incorrect usage of `input_type` can lead to a significant drop in retrieval performance.



All models listed [here](https://build.nvidia.com/explore/retrieval) are supported:

| Model Name         | Function Call                                         |
| :---               | :---                                                  |
| NV-Embed-QA | `embedding(model="nvidia_nim/NV-Embed-QA", input)` |
| nvidia/nv-embed-v1 | `embedding(model="nvidia_nim/nvidia/nv-embed-v1", input)` |
| nvidia/nv-embedqa-mistral-7b-v2 | `embedding(model="nvidia_nim/nvidia/nv-embedqa-mistral-7b-v2", input)` |
| nvidia/nv-embedqa-e5-v5 | `embedding(model="nvidia_nim/nvidia/nv-embedqa-e5-v5", input)` |
| nvidia/embed-qa-4 | `embedding(model="nvidia_nim/nvidia/embed-qa-4", input)` |
| nvidia/llama-3.2-nv-embedqa-1b-v1 | `embedding(model="nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v1", input)` |
| nvidia/llama-3.2-nv-embedqa-1b-v2 | `embedding(model="nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2", input)` |
| snowflake/arctic-embed-l | `embedding(model="nvidia_nim/snowflake/arctic-embed-l", input)` |
| baai/bge-m3 | `embedding(model="nvidia_nim/baai/bge-m3", input)` |


## HuggingFace Embedding Models
Dheera AI supports all Feature-Extraction + Sentence Similarity Embedding models: https://huggingface.co/models?pipeline_tag=feature-extraction

### Usage
```python
from dheera_ai import embedding
import os
os.environ['HUGGINGFACE_API_KEY'] = ""
response = embedding(
    model='huggingface/microsoft/codebert-base', 
    input=["good morning from dheera_ai"]
)
```

### Usage - Set input_type

Dheera AI infers input type (feature-extraction or sentence-similarity) by making a GET request to the api base. 

Override this, by setting the `input_type` yourself.

```python
from dheera_ai import embedding
import os
os.environ['HUGGINGFACE_API_KEY'] = ""
response = embedding(
    model='huggingface/microsoft/codebert-base', 
    input=["good morning from dheera_ai", "you are a good bot"],
    api_base = "https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud", 
    input_type="sentence-similarity"
)
```

### Usage - Custom API Base
```python
from dheera_ai import embedding
import os
os.environ['HUGGINGFACE_API_KEY'] = ""
response = embedding(
    model='huggingface/microsoft/codebert-base', 
    input=["good morning from dheera_ai"],
    api_base = "https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud"
)
```

| Model Name            | Function Call | Required OS Variables                        |
|-----------------------|--------------------------------------------------------------|-------------------------------------------------|
| microsoft/codebert-base    | `embedding('huggingface/microsoft/codebert-base', input=input)`               | `os.environ['HUGGINGFACE_API_KEY']`                                             |
| BAAI/bge-large-zh | `embedding('huggingface/BAAI/bge-large-zh', input=input)`         | `os.environ['HUGGINGFACE_API_KEY']`                                             |
| any-hf-embedding-model | `embedding('huggingface/hf-embedding-model', input=input)`         | `os.environ['HUGGINGFACE_API_KEY']`                                             |


## Mistral AI Embedding Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported

### Usage
```python
from dheera_ai import embedding
import os

os.environ['MISTRAL_API_KEY'] = ""
response = embedding(
    model="mistral/mistral-embed",
    input=["good morning from dheera_ai"],
)
print(response)
```

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mistral-embed | `embedding(model="mistral/mistral-embed", input)` | 

## Gemini AI Embedding Models

### API keys

This can be set as env variables or passed as **params to dheera_ai.embedding()**
```python
import os
os.environ["GEMINI_API_KEY"] = ""
```

### Usage - Embedding
```python
from dheera_ai import embedding
response = embedding(
  model="gemini/text-embedding-004",
  input=["good morning from dheera_ai"],
)
print(response)
```

All models listed [here](https://ai.google.dev/gemini-api/docs/models/gemini) are supported:

| Model Name         | Function Call                                         |
| :---               | :---                                                  |
| text-embedding-004 | `embedding(model="gemini/text-embedding-004", input)` |


## Vertex AI Embedding Models

### Usage - Embedding
```python
import dheera_ai
from dheera_ai import embedding
dheera_ai.vertex_project = "hardy-device-38811" # Your Project ID
dheera_ai.vertex_location = "us-central1"  # proj location

response = embedding(
    model="vertex_ai/textembedding-gecko",
    input=["good morning from dheera_ai"],
)
print(response)
```

### Supported Models
All models listed [here](https://github.com/BerriAI/dheera_ai/blob/57f37f743886a0249f630a6792d49dffc2c5d9b7/model_prices_and_context_window.json#L835) are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| textembedding-gecko | `embedding(model="vertex_ai/textembedding-gecko", input)` | 
| textembedding-gecko-multilingual | `embedding(model="vertex_ai/textembedding-gecko-multilingual", input)` | 
| textembedding-gecko-multilingual@001 | `embedding(model="vertex_ai/textembedding-gecko-multilingual@001", input)` | 
| textembedding-gecko@001 | `embedding(model="vertex_ai/textembedding-gecko@001", input)` | 
| textembedding-gecko@003 | `embedding(model="vertex_ai/textembedding-gecko@003", input)` | 
| text-embedding-preview-0409 | `embedding(model="vertex_ai/text-embedding-preview-0409", input)` |
| text-multilingual-embedding-preview-0409 | `embedding(model="vertex_ai/text-multilingual-embedding-preview-0409", input)` | 

## Voyage AI Embedding Models

### Usage - Embedding
```python
from dheera_ai import embedding
import os

os.environ['VOYAGE_API_KEY'] = ""
response = embedding(
    model="voyage/voyage-01",
    input=["good morning from dheera_ai"],
)
print(response)
```

### Supported Models
All models listed here https://docs.voyageai.com/embeddings/#models-and-specifics are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| voyage-01 | `embedding(model="voyage/voyage-01", input)` | 
| voyage-lite-01 | `embedding(model="voyage/voyage-lite-01", input)` | 
| voyage-lite-01-instruct | `embedding(model="voyage/voyage-lite-01-instruct", input)` | 

### Provider-specific Params


:::info

Any non-openai params, will be treated as provider-specific params, and sent in the request body as kwargs to the provider.

[**See Reserved Params**](https://github.com/BerriAI/dheera_ai/blob/2f5f85cb52f36448d1f8bbfbd3b8af8167d0c4c8/dheera_ai/main.py#L3130)
:::

### **Example**

Cohere v3 Models have a required parameter: `input_type`, it can be one of the following four values:

- `input_type="search_document"`: (default) Use this for texts (documents) you want to store in your vector database
- `input_type="search_query"`: Use this for search queries to find the most relevant documents in your vector database
- `input_type="classification"`: Use this if you use the embeddings as an input for a classification system
- `input_type="clustering"`: Use this if you use the embeddings for text clustering

https://txt.cohere.com/introducing-embed-v3/

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from dheera_ai import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from dheera_ai", "this is another item"], 
    input_type="search_document" # 👈 PROVIDER-SPECIFIC PARAM
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**via config**

```yaml
model_list:
  - model_name: "cohere-embed"
    dheera_ai_params:
      model: embed-english-v3.0
      input_type: search_document # 👈 PROVIDER-SPECIFIC PARAM
```

**via request**

```bash
curl -X POST 'http://0.0.0.0:4000/v1/embeddings' \
-H 'Authorization: Bearer sk-54d77cd67b9febbb' \
-H 'Content-Type: application/json' \
-d '{
  "model": "cohere-embed",
  "input": ["Are you authorized to work in United States of America?"],
  "input_type": "search_document" # 👈 PROVIDER-SPECIFIC PARAM
}'
```
</TabItem>
</Tabs>

## Nebius AI Studio Embedding Models

### Usage - Embedding
```python
from dheera_ai import embedding
import os

os.environ['NEBIUS_API_KEY'] = ""
response = embedding(
    model="nebius/BAAI/bge-en-icl",
    input=["Good morning from dheera_ai!"],
)
print(response)
```

### Supported Models
All supported models can be found here: https://studio.nebius.ai/models/embedding

| Model Name               | Function Call                                                   |
|--------------------------|-----------------------------------------------------------------|
| BAAI/bge-en-icl | `embedding(model="nebius/BAAI/bge-en-icl", input)`              | 
| BAAI/bge-multilingual-gemma2 | `embedding(model="nebius/BAAI/bge-multilingual-gemma2", input)` | 
| intfloat/e5-mistral-7b-instruct | `embedding(model="nebius/intfloat/e5-mistral-7b-instruct", input)`      | 

