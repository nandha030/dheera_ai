import Image from '@theme/IdealImage';

# Logfire

Logfire is open Source Observability & Analytics for LLM Apps
Detailed production traces and a granular view on quality, cost and latency

<Image img={require('../../img/logfire.png')} />

:::info
We want to learn how we can make the callbacks better! Meet the Dheera AI [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-dheera_ai-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

Ensure you have installed the following packages to use this integration

```shell
pip install dheera_ai

pip install opentelemetry-api==1.25.0
pip install opentelemetry-sdk==1.25.0
pip install opentelemetry-exporter-otlp==1.25.0
```

## Quick Start

Get your Logfire token from [Logfire](https://logfire.pydantic.dev/)

```python
dheera_ai.callbacks = ["logfire"]
```

```python
# pip install logfire
import dheera_ai
import os

# from https://logfire.pydantic.dev/
os.environ["LOGFIRE_TOKEN"] = ""

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set logfire as a callback, dheera_ai will send the data to logfire
dheera_ai.success_callback = ["logfire"]

# openai call
response = dheera_ai.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-dheera_ai-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
