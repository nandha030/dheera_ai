
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';

# Provider Files Endpoints

Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

Use this to call the provider's `/files` endpoints directly, in the OpenAI format. 

## Quick Start

- Upload a File
- List Files
- Retrieve File Information
- Delete File
- Get File Content

## Multi-Account Support (Multiple OpenAI Keys)

Use different OpenAI API keys for files and batches by specifying a `model` parameter that references entries in your `model_list`. This approach works **without requiring a database** and allows you to route files/batches to different OpenAI accounts.

### How It Works

1. Define models in `model_list` with different API keys
2. Pass `model` parameter when creating files
3. Dheera AI returns encoded IDs that contain routing information
4. Use encoded IDs for all subsequent operations (retrieve, delete, batches)
5. No need to specify model again - routing info is in the ID

### Setup

```yaml
model_list:
  # dheera_ai OpenAI Account
  - model_name: "gpt-4o-dheera_ai"
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_DHEERA_AI_API_KEY
  
  # Free OpenAI Account
  - model_name: "gpt-4o-free"
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_FREE_API_KEY
```

### Usage Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",  # Your Dheera AI proxy key
    base_url="http://0.0.0.0:4000"
)

# Create file using dheera_ai account
file_response = client.files.create(
    file=open("batch_data.jsonl", "rb"),
    purpose="batch",
    extra_body={"model": "gpt-4o-dheera_ai"}  # Routes to dheera_ai key
)
print(f"File ID: {file_response.id}")
# Returns encoded ID like: file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8taWZvb2Q

# Create batch using the encoded file ID
# No need to specify model again - it's embedded in the file ID
batch_response = client.batches.create(
    input_file_id=file_response.id,  # Encoded ID
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch ID: {batch_response.id}")
# Returns encoded batch ID with routing info

# Retrieve batch - routing happens automatically
batch_status = client.batches.retrieve(batch_response.id)
print(f"Status: {batch_status.status}")

# List files for a specific account
files = client.files.list(
    extra_body={"model": "gpt-4o-free"}  # List free files
)

# List batches for a specific account
batches = client.batches.list(
    extra_query={"model": "gpt-4o-dheera_ai"}  # List dheera_ai batches
)
```

### Parameter Options

You can pass the `model` parameter via:
- **Request body**: `extra_body={"model": "gpt-4o-dheera_ai"}`
- **Query parameter**: `?model=gpt-4o-dheera_ai`
- **Header**: `x-dheera_ai-model: gpt-4o-dheera_ai`

### How Encoded IDs Work

- When you create a file/batch with a `model` parameter, Dheera AI encodes the model name into the returned ID
- The encoded ID is base64-encoded and looks like: `file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8taWZvb2Q`
- When you use this ID in subsequent operations (retrieve, delete, batch create), Dheera AI automatically:
  1. Decodes the ID
  2. Extracts the model name
  3. Looks up the credentials
  4. Routes the request to the correct OpenAI account
- The original provider file/batch ID is preserved internally

### Benefits

✅ **No Database Required** - All routing info stored in the ID  
✅ **Stateless** - Works across proxy restarts  
✅ **Simple** - Just pass the ID around like normal  
✅ **Backward Compatible** - Existing `custom_llm_provider` and `files_settings` still work  
✅ **Future-Proof** - Aligns with managed batches approach  

### Migration from files_settings

**Old approach (still works):**
```yaml
files_settings:
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_KEY
```

```python
# Had to specify provider on every call
client.files.create(..., extra_headers={"custom-llm-provider": "openai"})
client.files.retrieve(file_id, extra_headers={"custom-llm-provider": "openai"})
```

**New approach (recommended):**
```yaml
model_list:
  - model_name: "gpt-4o-account1"
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_KEY
```

```python
# Specify model once on create
file = client.files.create(..., extra_body={"model": "gpt-4o-account1"})

# Then just use the ID - routing is automatic
client.files.retrieve(file.id)  # No need to specify account
client.batches.create(input_file_id=file.id)  # Routes correctly
```

<Tabs>
<TabItem value="proxy" label="Dheera AI PROXY Server">

1. Setup config.yaml

```
# for /files endpoints
files_settings:
  - custom_llm_provider: azure
    api_base: https://exampleopenaiendpoint-production.up.railway.app
    api_key: fake-key
    api_version: "2023-03-15-preview"
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_API_KEY
```

2. Start Dheera AI PROXY Server

```bash
dheera_ai --config /path/to/config.yaml

## RUNNING on http://0.0.0.0:4000
```

3. Use OpenAI's /files endpoints

Upload a File

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

client.files.create(
    file=wav_data,
    purpose="user_data",
    extra_headers={"custom-llm-provider": "openai"}
)
```

List Files

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

files = client.files.list(extra_headers={"custom-llm-provider": "openai"})
print("files=", files)
```

Retrieve File Information

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

file = client.files.retrieve(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
print("file=", file)
```

Delete File

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

response = client.files.delete(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
print("delete response=", response)
```

Get File Content

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://0.0.0.0:4000/v1"
)

content = client.files.content(file_id="file-abc123", extra_headers={"custom-llm-provider": "openai"})
print("content=", content)
```

</TabItem>
<TabItem value="sdk" label="SDK">

**Upload a File**
```python
from dheera_ai
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

file_obj = await dheera_ai.acreate_file(
    file=open("mydata.jsonl", "rb"),
    purpose="fine-tune",
    custom_llm_provider="openai",
)
print("Response from creating file=", file_obj)
```

**List Files**
```python
files = await dheera_ai.alist_files(
    custom_llm_provider="openai",
    limit=10
)
print("files=", files)
```

**Retrieve File Information**
```python
file = await dheera_ai.aretrieve_file(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("file=", file)
```

**Delete File**
```python
response = await dheera_ai.adelete_file(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("delete response=", response)
```

**Get File Content**
```python
content = await dheera_ai.afile_content(
    file_id="file-abc123",
    custom_llm_provider="openai"
)
print("file content=", content)
```

**Get File Content (Bedrock)**
```python
# For Bedrock batch output files stored in S3
content = await dheera_ai.afile_content(
    file_id="s3://bucket-name/path/to/file.jsonl",  # S3 URI or unified file ID
    custom_llm_provider="bedrock",
    aws_region_name="us-west-2"
)
print("file content=", content.text)
```

</TabItem>
</Tabs>


## **Supported Providers**:

### [OpenAI](#quick-start)

### [Azure OpenAI](./providers/azure#azure-batches-api)

### [Vertex AI](./providers/vertex#batch-apis)

### [Bedrock](./providers/bedrock_batches#4-retrieve-batch-results)

## [Swagger API Reference](https://dheera_ai-api.up.railway.app/#/files)
