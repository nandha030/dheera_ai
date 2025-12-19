import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Adding a New Guardrail Integration

You're going to create a class that checks text before it goes to the LLM or after it comes back. If it violates your rules, you block it.

## How It Works

Request with guardrail:

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "How do I hack a system?"}],
    "guardrails": ["my-guardrail"]
}'
```

Your guardrail checks input, then output. If something's wrong, raise an exception.

## Build Your Guardrail

### Create Your Directory

```bash
mkdir -p dheera_ai/proxy/guardrails/guardrail_hooks/my_guardrail
cd dheera_ai/proxy/guardrails/guardrail_hooks/my_guardrail
```

Two files: `my_guardrail.py` (main class) and `__init__.py` (initialization).

### Write the Main Class

`my_guardrail.py`:

Follow from [Custom Guardrail](../proxy/guardrails/custom_guardrail#custom-guardrail) tutorial.

### Create the Init File

`__init__.py`:

```python
from typing import TYPE_CHECKING

from dheera_ai.types.guardrails import SupportedGuardrailIntegrations

from .my_guardrail import MyGuardrail

if TYPE_CHECKING:
    from dheera_ai.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(dheera_ai_params: "LitellmParams", guardrail: "Guardrail"):
    import dheera_ai
    
    _my_guardrail_callback = MyGuardrail(
        api_base=dheera_ai_params.api_base,
        api_key=dheera_ai_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=dheera_ai_params.mode,
        default_on=dheera_ai_params.default_on,
    )
    
    dheera_ai.logging_callback_manager.add_dheera_ai_callback(_my_guardrail_callback)
    return _my_guardrail_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MY_GUARDRAIL.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MY_GUARDRAIL.value: MyGuardrail,
}
```

### Register Your Guardrail Type

Add to `dheera_ai/types/guardrails.py`:

```python
class SupportedGuardrailIntegrations(str, Enum):
    LAKERA = "lakera_prompt_injection"
    APORIA = "aporia"
    BEDROCK = "bedrock_guardrails"
    PRESIDIO = "presidio"
    ZSCALER_AI_GUARD = "zscaler_ai_guard"
    MY_GUARDRAIL = "my_guardrail"
```

## Usage

### Config File

```yaml
model_list:
  - model_name: gpt-4
    dheera_ai_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

dheera_ai_settings:
  guardrails:
    - guardrail_name: my_guardrail
      dheera_ai_params:
        guardrail: my_guardrail
        mode: during_call
        api_key: os.environ/MY_GUARDRAIL_API_KEY
        api_base: https://api.myguardrail.com
```

### Per-Request

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Test message"}],
    "guardrails": ["my_guardrail"]
}'
```

## Testing

Add unit tests inside `test_dheera_ai/` folder.



