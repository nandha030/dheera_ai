import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Triton Inference Server

Dheera AI supports Embedding Models on Triton Inference Servers

| Property | Details |
|-------|-------|
| Description | NVIDIA Triton Inference Server |
| Provider Route on Dheera AI | `triton/` |
| Supported Operations | `/chat/completion`, `/completion`, `/embedding` |
| Supported Triton endpoints | `/infer`, `/generate`, `/embeddings` |
| Link to Provider Doc | [Triton Inference Server ↗](https://developer.nvidia.com/triton-inference-server) |

## Triton `/generate` - Chat Completion 


<Tabs>
<TabItem value="sdk" label="SDK">

Use the `triton/` prefix to route to triton server
```python
from dheera_ai import completion
response = completion(
    model="triton/llama-3-8b-instruct",
    messages=[{"role": "user", "content": "who are u?"}],
    max_tokens=10,
    api_base="http://localhost:8000/generate",
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: my-triton-model
      dheera_ai_params:
        model: triton/<your-triton-model>"
        api_base: https://your-triton-api-base/triton/generate
  ```


2. Start the proxy 

  ```bash
  $ dheera_ai --config /path/to/config.yaml --detailed_debug
  ```

3. Send Request to Dheera AI Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

    ```python
    import openai
    from openai import OpenAI

    # set base_url to your proxy server
    # set api_key to send to proxy server
    client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

    response = client.chat.completions.create(
        model="my-triton-model",
        messages=[{"role": "user", "content": "who are u?"}],
        max_tokens=10,
    )

    print(response)

    ```

  </TabItem>

  <TabItem value="curl" label="curl">

  `--header` is optional, only required if you're using dheera_ai proxy with Virtual Keys

    ```shell
    curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data ' {
    "model": "my-triton-model",
    "messages": [{"role": "user", "content": "who are u?"}]
    }'

    ```
  </TabItem>

  </Tabs>

</TabItem>
</Tabs>

## Triton `/infer` - Chat Completion 

<Tabs>
<TabItem value="sdk" label="SDK">


Use the `triton/` prefix to route to triton server
```python
from dheera_ai import completion


response = completion(
    model="triton/llama-3-8b-instruct",
    messages=[{"role": "user", "content": "who are u?"}],
    max_tokens=10,
    api_base="http://localhost:8000/infer",
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: my-triton-model
      dheera_ai_params:
        model: triton/<your-triton-model>"
        api_base: https://your-triton-api-base/triton/infer
  ```


2. Start the proxy 

  ```bash
  $ dheera_ai --config /path/to/config.yaml --detailed_debug
  ```

3. Send Request to Dheera AI Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

    ```python
    import openai
    from openai import OpenAI

    # set base_url to your proxy server
    # set api_key to send to proxy server
    client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

    response = client.chat.completions.create(
        model="my-triton-model",
        messages=[{"role": "user", "content": "who are u?"}],
        max_tokens=10,
    )

    print(response)

    ```

  </TabItem>

  <TabItem value="curl" label="curl">

  `--header` is optional, only required if you're using dheera_ai proxy with Virtual Keys

    ```shell
    curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data ' {
    "model": "my-triton-model",
    "messages": [{"role": "user", "content": "who are u?"}]
    }'

    ```
  </TabItem>

  </Tabs>

</TabItem>
</Tabs>



## Triton `/embeddings` - Embedding

<Tabs>
<TabItem value="sdk" label="SDK">

Use the `triton/` prefix to route to triton server
```python
from dheera_ai import embedding
import os

response = await dheera_ai.aembedding(
    model="triton/<your-triton-model>",                                                       
    api_base="https://your-triton-api-base/triton/embeddings", # /embeddings endpoint you want dheera_ai to call on your server
    input=["good morning from dheera_ai"],
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: my-triton-model
      dheera_ai_params:
        model: triton/<your-triton-model>"
        api_base: https://your-triton-api-base/triton/embeddings
  ```


2. Start the proxy 

  ```bash
  $ dheera_ai --config /path/to/config.yaml --detailed_debug
  ```

3. Send Request to Dheera AI Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

    ```python
    import openai
    from openai import OpenAI

    # set base_url to your proxy server
    # set api_key to send to proxy server
    client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

    response = client.embeddings.create(
        input=["hello from dheera_ai"],
        model="my-triton-model"
    )

    print(response)

    ```

  </TabItem>

  <TabItem value="curl" label="curl">

  `--header` is optional, only required if you're using dheera_ai proxy with Virtual Keys

    ```shell
    curl --location 'http://0.0.0.0:4000/embeddings' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data ' {
    "model": "my-triton-model",
    "input": ["write a dheera_ai poem"]
    }'

    ```
  </TabItem>

  </Tabs>


</TabItem>

</Tabs>
