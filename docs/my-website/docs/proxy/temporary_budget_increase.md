# ✨ Temporary Budget Increase

Set temporary budget increase for a Dheera AI Virtual Key. Use this if you get asked to increase the budget for a key temporarily.


| Hierarchy | Supported | 
|-----------|-----------|
| Dheera AI Virtual Key | ✅ |
| User | ❌ |
| Team | ❌ |
| Organization | ❌ |

:::note

✨ Temporary Budget Increase is a Dheera AI Enterprise feature.

[Enterprise Pricing](https://www.dheera_ai.ai/#pricing)

[Get free 7-day trial key](https://www.dheera_ai.ai/enterprise#trial)

:::


1. Create a Dheera AI Virtual Key with budget

```bash
curl -L -X POST 'http://localhost:4000/key/generate' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer DHEERA_AI_MASTER_KEY' \
-d '{
    "max_budget": 0.0000001
}'
```

Expected response:

```json
{
    "key": "sk-your-new-key"
}
```

2. Update key with temporary budget increase

```bash
curl -L -X POST 'http://localhost:4000/key/update' \
-H 'Authorization: Bearer DHEERA_AI_MASTER_KEY' \
-H 'Content-Type: application/json' \
-d '{
    "key": "sk-your-new-key",
    "temp_budget_increase": 100, 
    "temp_budget_expiry": "2025-01-15"
}'
```

3. Test it! 

```bash
curl -L -X POST 'http://localhost:4000/chat/completions' \
-H 'Authorization: Bearer sk-your-new-key' \
-H 'Content-Type: application/json' \
-d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, world!"}]
}'
```

Expected Response Header:

```
x-dheera_ai-key-max-budget: 100.0000001
```


