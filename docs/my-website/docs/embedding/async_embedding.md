# dheera_ai.aembedding()

Dheera AI provides an asynchronous version of the `embedding` function called `aembedding`
### Usage
```python
from dheera_ai import aembedding
import asyncio

async def test_get_response():
    response = await aembedding('text-embedding-ada-002', input=["good morning from dheera_ai"])
    return response

response = asyncio.run(test_get_response())
print(response)
```