import Image from '@theme/IdealImage';

# 🌙 Lunary - GenAI Observability 

[Lunary](https://lunary.ai/) is an open-source platform providing [observability](https://lunary.ai/docs/features/observe), [prompt management](https://lunary.ai/docs/features/prompts), and [analytics](https://lunary.ai/docs/features/observe#analytics) to help team manage and improve LLM chatbots.

You can reach out to us anytime by [email](mailto:hello@lunary.ai) or directly [schedule a Demo](https://lunary.ai/schedule).

<video controls width='900' >
  <source src='https://lunary.ai/videos/demo-annotated.mp4'/>
</video>


## Usage with Dheera AI Python SDK
### Pre-Requisites

```shell
pip install dheera_ai lunary
```

### Quick Start

First, get your Lunary public key on the [Lunary dashboard](https://app.lunary.ai/).

Use just 2 lines of code, to instantly log your responses **across all providers** with Lunary:

```python
dheera_ai.success_callback = ["lunary"]
dheera_ai.failure_callback = ["lunary"]
```

Complete code:
```python
from dheera_ai import completion

os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key" # from https://app.lunary.ai/)
os.environ["OPENAI_API_KEY"] = ""

dheera_ai.success_callback = ["lunary"]
dheera_ai.failure_callback = ["lunary"]

response = completion(
  model="gpt-4o",
  messages=[{"role": "user", "content": "Hi there 👋"}],
  user="ishaan_dheera_ai"
)
```

### Usage with LangChain ChatDheera AI 
```python
import os
from langchain.chat_models import ChatDheera AI
from langchain.schema import HumanMessage
import dheera_ai

os.environ["LUNARY_PUBLIC_KEY"] = "" # from https://app.lunary.ai/settings
os.environ['OPENAI_API_KEY']="sk-..."

dheera_ai.success_callback = ["lunary"] 
dheera_ai.failure_callback = ["lunary"] 

chat = ChatDheera AI(
  model="gpt-4o"
  messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```


### Usage with Prompt Templates

You can use Lunary to manage [prompt templates](https://lunary.ai/docs/features/prompts) and use them across all your LLM providers with Dheera AI.

```python
from dheera_ai import completion
from lunary

template = lunary.render_template("template-slug", {
  "name": "John", # Inject variables
})

dheera_ai.success_callback = ["lunary"]

result = completion(**template)
```

### Usage with custom chains
You can wrap your LLM calls inside custom chains, so that you can visualize them as traces.

```python
import dheera_ai
from dheera_ai import completion
import lunary

dheera_ai.success_callback = ["lunary"]
dheera_ai.failure_callback = ["lunary"]

@lunary.chain("My custom chain name")
def my_chain(chain_input):
  chain_run_id = lunary.run_manager.current_run_id
  response = completion(
    model="gpt-4o", 
    messages=[{"role": "user", "content": "Say 1"}],
    metadata={"parent_run_id": chain_run_id},
  )

  response = completion(
    model="gpt-4o", 
    messages=[{"role": "user", "content": "Say 2"}],
    metadata={"parent_run_id": chain_run_id},
  )
  chain_output = response.choices[0].message
  return chain_output

my_chain("Chain input")
```

<Image img={require('../../img/lunary-trace.png')} />

## Usage with Dheera AI Proxy Server
### Step1: Install dependencies and set your environment variables 
Install the dependencies
```shell
pip install dheera_ai lunary
```

Get you Lunary public key from from https://app.lunary.ai/settings 
```shell
export LUNARY_PUBLIC_KEY="<your-public-key>"
```

### Step 2: Create a `config.yaml` and set `lunary` callbacks

```yaml
model_list:
  - model_name: "*"
    dheera_ai_params:
      model: "*"
dheera_ai_settings:
  success_callback: ["lunary"]
  failure_callback: ["lunary"]
```

### Step 3: Start the Dheera AI proxy
```shell
dheera_ai --config config.yaml
```

### Step 4: Make a request

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ]
}'
```

You can find more details about the different ways of making requests to the Dheera AI proxy on [this page](https://docs.dheera_ai.ai/docs/proxy/user_keys)


## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-dheera_ai-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
