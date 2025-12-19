import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Secret Manager

Integrate your custom secret management system with Dheera AI.

## Quick Start

### 1. Create Your Secret Manager Class

Create a new file `my_secret_manager.py` with an in-memory secret store:

```python showLineNumbers title="my_secret_manager.py"
from typing import Optional, Union
import httpx
from dheera_ai.integrations.custom_secret_manager import CustomSecretManager

class InMemorySecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="in_memory_secrets")
        # Store your secrets in memory
        self.secrets = {
            "OPENAI_API_KEY": "sk-...",
            "ANTHROPIC_API_KEY": "sk-ant-...",
        }

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read secret asynchronously"""
        return self.secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read secret synchronously"""
        return self.secrets.get(secret_name)
```

### 2. Configure Proxy

Reference your custom secret manager in `config.yaml`:

```yaml showLineNumbers title="config.yaml"
general_settings:
  master_key: os.environ/DHEERA_AI_MASTER_KEY
  key_management_system: custom  # 👈 KEY CHANGE
  key_management_settings:
    custom_secret_manager: my_secret_manager.InMemorySecretManager  # 👈 KEY CHANGE

model_list:
  - model_name: gpt-4
    dheera_ai_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY  # Read from custom secret manager
```

### 3. Start Dheera AI Proxy

<Tabs>
<TabItem value="docker" label="Docker">

Mount your custom secret manager file on the container:

```bash showLineNumbers
docker run -d \
  -p 4000:4000 \
  -e DHEERA_AI_MASTER_KEY=$DHEERA_AI_MASTER_KEY \
  --name dheera_ai-proxy \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/my_secret_manager.py:/app/my_secret_manager.py \
  docker.dheera_ai.ai/berriai/dheera_ai:main-latest \
  --config /app/config.yaml \
  --port 4000 \
  --detailed_debug
```

</TabItem>

<TabItem value="pip" label="Python Package">

```bash
dheera_ai --config config.yaml --detailed_debug
```

</TabItem>
</Tabs>

## Configuration Options

Customize secret manager behavior in your `config.yaml`:

<Tabs>
<TabItem value="read_only" label="Read Keys Only">

```yaml showLineNumbers title="config.yaml"
general_settings:
  key_management_system: custom
  key_management_settings:
    custom_secret_manager: my_secret_manager.InMemorySecretManager
    hosted_keys: ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]  # Only check these keys
```

</TabItem>

<TabItem value="write_only" label="Store Virtual Keys">

Store Dheera AI proxy virtual keys in your secret manager:

```yaml showLineNumbers title="config.yaml"
general_settings:
  key_management_system: custom
  key_management_settings:
    custom_secret_manager: my_secret_manager.InMemorySecretManager
    access_mode: "write_only"
    store_virtual_keys: true
    prefix_for_stored_virtual_keys: "dheera_ai/"
    description: "Dheera AI virtual key"
    tags:
      Environment: "Production"
      Team: "AI"
```

</TabItem>

<TabItem value="read_and_write" label="Read + Write">

```yaml showLineNumbers title="config.yaml"
general_settings:
  key_management_system: custom
  key_management_settings:
    custom_secret_manager: my_secret_manager.InMemorySecretManager
    access_mode: "read_and_write"
    hosted_keys: ["OPENAI_API_KEY"]
    store_virtual_keys: true
    prefix_for_stored_virtual_keys: "dheera_ai/"
```

</TabItem>
</Tabs>

### Available Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `custom_secret_manager` | Path to your custom secret manager class | Required |
| `access_mode` | `"read_only"`, `"write_only"`, or `"read_and_write"` | `"read_only"` |
| `hosted_keys` | List of specific keys to check in secret manager | All keys |
| `store_virtual_keys` | Store Dheera AI virtual keys in secret manager | `false` |
| `prefix_for_stored_virtual_keys` | Prefix for stored virtual keys | `"dheera_ai/"` |
| `description` | Description for stored secrets | `None` |
| `tags` | Tags to apply to stored secrets | `None` |

## Required Methods

Your custom secret manager **must** implement these two methods:

### `async_read_secret()`

```python showLineNumbers
async def async_read_secret(
    self,
    secret_name: str,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Optional[str]:
    """
    Read a secret asynchronously.
    
    Returns:
        Secret value if found, None otherwise
    """
    pass
```

### `sync_read_secret()`

```python showLineNumbers
def sync_read_secret(
    self,
    secret_name: str,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Optional[str]:
    """
    Read a secret synchronously.
    
    Returns:
        Secret value if found, None otherwise
    """
    pass
```

## Optional Methods

Implement these for additional functionality:

### `async_write_secret()`

```python showLineNumbers
async def async_write_secret(
    self,
    secret_name: str,
    secret_value: str,
    description: Optional[str] = None,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    tags: Optional[Union[dict, list]] = None,
) -> dict:
    """Write a secret to your secret manager"""
    pass
```

### `async_delete_secret()`

```python showLineNumbers
async def async_delete_secret(
    self,
    secret_name: str,
    recovery_window_in_days: Optional[int] = 7,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> dict:
    """Delete a secret from your secret manager"""
    pass
```

## Use Cases

✅ Proprietary vault systems  
✅ Custom authentication (mTLS, OAuth)  
✅ Organization-specific security policies  
✅ Legacy secret storage systems  
✅ Multi-region secret replication  
✅ Secret versioning and rotation  
✅ Compliance requirements (HIPAA, SOC2)  

## Example

See [cookbook/dheera_ai_proxy_server/secret_manager/my_secret_manager.py](https://github.com/BerriAI/dheera_ai/blob/main/cookbook/dheera_ai_proxy_server/secret_manager/my_secret_manager.py) for a complete working example with:

- In-memory secret manager implementation  
- Integration with Dheera AI Proxy  
- Read, write, and delete operations

