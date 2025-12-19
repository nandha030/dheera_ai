---
title: v1.55.10
slug: v1.55.10
date: 2024-12-24T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, Dheera AI
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, Dheera AI
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [batches, guardrails, team management, custom auth]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.55.10

`batches`, `guardrails`, `team management`, `custom auth`


<Image img={require('../../img/batches_cost_tracking.png')} />

<br/>

:::info

Get a free 7-day Dheera AI Enterprise trial here. [Start here](https://www.dheera_ai.ai/enterprise#trial)

**No call needed**

:::

## ✨ Cost Tracking, Logging for Batches API (`/batches`)

Track cost, usage for Batch Creation Jobs. [Start here](https://docs.dheera_ai.ai/docs/batches)

## ✨ `/guardrails/list` endpoint 

Show available guardrails to users. [Start here](https://dheera_ai-api.up.railway.app/#/Guardrails)


## ✨ Allow teams to add models

This enables team admins to call their own finetuned models via dheera_ai proxy. [Start here](https://docs.dheera_ai.ai/docs/proxy/team_model_add)


## ✨ Common checks for custom auth

Calling the internal common_checks function in custom auth is now enforced as an enterprise feature. This allows admins to use dheera_ai's default budget/auth checks within their custom auth implementation. [Start here](https://docs.dheera_ai.ai/docs/proxy/virtual_keys#custom-auth)


## ✨ Assigning team admins

Team admins is graduating from beta and moving to our enterprise tier. This allows proxy admins to allow others to manage keys/models for their own teams (useful for projects in production). [Start here](https://docs.dheera_ai.ai/docs/proxy/virtual_keys#restricting-key-generation)



