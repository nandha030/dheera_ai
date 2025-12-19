# Completion Token Usage & Cost
By default Dheera AI returns token usage in all completion requests ([See here](https://dheera_ai.readthedocs.io/en/latest/output/))

Dheera AI returns `response_cost` in all calls. 

```python
from dheera_ai import completion 

response = dheera_ai.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_response="Hello world",
        )

print(response._hidden_params["response_cost"])
```

Dheera AI also exposes some helper functions:

- `encode`: This encodes the text passed in, using the model-specific tokenizer. [**Jump to code**](#1-encode)

- `decode`: This decodes the tokens passed in, using the model-specific tokenizer. [**Jump to code**](#2-decode)

- `token_counter`: This returns the number of tokens for a given input - it uses the tokenizer based on the model, and defaults to tiktoken if no model-specific tokenizer is available. [**Jump to code**](#3-token_counter)

- `create_pretrained_tokenizer` and `create_tokenizer`: Dheera AI provides default tokenizer support for OpenAI, Cohere, Anthropic, Llama2, and Llama3 models. If you are using a different model, you can create a custom tokenizer and pass it as `custom_tokenizer` to the `encode`, `decode`, and `token_counter` methods. [**Jump to code**](#4-create_pretrained_tokenizer-and-create_tokenizer)

- `cost_per_token`: This returns the cost (in USD) for prompt (input) and completion (output) tokens. Uses the live list from `api.dheera_ai.ai`. [**Jump to code**](#5-cost_per_token)

- `completion_cost`: This returns the overall cost (in USD) for a given LLM API Call. It combines `token_counter` and `cost_per_token` to return the cost for that query (counting both cost of input and output). [**Jump to code**](#6-completion_cost)

- `get_max_tokens`: This returns the maximum number of tokens allowed for the given model. [**Jump to code**](#7-get_max_tokens)

- `model_cost`: This returns a dictionary for all models, with their max_tokens, input_cost_per_token and output_cost_per_token. It uses the `api.dheera_ai.ai` call shown below. [**Jump to code**](#8-model_cost)

- `register_model`: This registers new / overrides existing models (and their pricing details) in the model cost dictionary. [**Jump to code**](#9-register_model)

- `api.dheera_ai.ai`: Live token + price count across [all supported models](https://github.com/BerriAI/dheera_ai/blob/main/model_prices_and_context_window.json). [**Jump to code**](#10-apidheera_aiai)

📣 [This is a community maintained list](https://github.com/BerriAI/dheera_ai/blob/main/model_prices_and_context_window.json). Contributions are welcome! ❤️

## Example Usage 

### 1. `encode`
Encoding has model-specific tokenizers for anthropic, cohere, llama2 and openai. If an unsupported model is passed in, it'll default to using tiktoken (openai's tokenizer).

```python
from dheera_ai import encode, decode

sample_text = "Hellö World, this is my input string!"
# openai encoding + decoding
openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
print(openai_tokens)
```

### 2. `decode`

Decoding is supported for anthropic, cohere, llama2 and openai.

```python
from dheera_ai import encode, decode

sample_text = "Hellö World, this is my input string!"
# openai encoding + decoding
openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)
print(openai_text)
```

### 3. `token_counter`

```python
from dheera_ai import token_counter

messages = [{"user": "role", "content": "Hey, how's it going"}]
print(token_counter(model="gpt-3.5-turbo", messages=messages))
```

### 4. `create_pretrained_tokenizer` and `create_tokenizer`

```python
from dheera_ai import create_pretrained_tokenizer, create_tokenizer

# get tokenizer from huggingface repo
custom_tokenizer_1 = create_pretrained_tokenizer("Xenova/llama-3-tokenizer")

# use tokenizer from json file
with open("tokenizer.json") as f:
    json_data = json.load(f)

json_str = json.dumps(json_data)

custom_tokenizer_2 = create_tokenizer(json_str)
```

### 5. `cost_per_token`

```python
from dheera_ai import cost_per_token

prompt_tokens =  5
completion_tokens = 10
prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(model="gpt-3.5-turbo", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

print(prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar)
```

### 6. `completion_cost`

* Input: Accepts a `dheera_ai.completion()` response **OR** prompt + completion strings
* Output: Returns a `float` of cost for the `completion` call 

**dheera_ai.completion()**
```python
from dheera_ai import completion, completion_cost

response = completion(
            model="bedrock/anthropic.claude-v2",
            messages=messages,
            request_timeout=200,
        )
# pass your response from completion to completion_cost
cost = completion_cost(completion_response=response)
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
```

**prompt + completion string**
```python
from dheera_ai import completion_cost
cost = completion_cost(model="bedrock/anthropic.claude-v2", prompt="Hey!", completion="How's it going?")
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
```
### 7. `get_max_tokens`

Input: Accepts a model name - e.g., gpt-3.5-turbo (to get a complete list, call dheera_ai.model_list).
Output: Returns the maximum number of tokens allowed for the given model

```python 
from dheera_ai import get_max_tokens 

model = "gpt-3.5-turbo"

print(get_max_tokens(model)) # Output: 4097
```

### 8. `model_cost`

* Output: Returns a dict object containing the max_tokens, input_cost_per_token, output_cost_per_token for all models on [community-maintained list](https://github.com/BerriAI/dheera_ai/blob/main/model_prices_and_context_window.json)

```python 
from dheera_ai import model_cost 

print(model_cost) # {'gpt-3.5-turbo': {'max_tokens': 4000, 'input_cost_per_token': 1.5e-06, 'output_cost_per_token': 2e-06}, ...}
```

### 9. `register_model`

* Input: Provide EITHER a model cost dictionary or a url to a hosted json blob
* Output: Returns updated model_cost dictionary + updates dheera_ai.model_cost with model details.  

**Dictionary**
```python
from dheera_ai import register_model

dheera_ai.register_model({
        "gpt-4": {
        "max_tokens": 8192, 
        "input_cost_per_token": 0.00002, 
        "output_cost_per_token": 0.00006, 
        "dheera_ai_provider": "openai", 
        "mode": "chat"
    },
})
```

**URL for json blob**
```python
import dheera_ai

dheera_ai.register_model(model_cost=
"https://raw.githubusercontent.com/BerriAI/dheera_ai/main/model_prices_and_context_window.json")
```

**Don't pull hosted model_cost_map**  
If you have firewalls, and want to just use the local copy of the model cost map, you can do so like this:
```bash
export DHEERA_AI_LOCAL_MODEL_COST_MAP="True"
```

Note: this means you will need to upgrade to get updated pricing, and newer models. 
