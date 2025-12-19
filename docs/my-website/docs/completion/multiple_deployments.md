# Multiple Deployments

If you have multiple deployments of the same model, you can pass the list of deployments, and Dheera AI will return the first result. 

## Quick Start

Multiple providers offer Mistral-7B-Instruct. 

Here's how you can use dheera_ai to return the first result: 

```python
from dheera_ai import completion

messages=[{"role": "user", "content": "Hey, how's it going?"}]

## All your mistral deployments ##
model_list = [{
	"model_name": "mistral-7b-instruct",
	"dheera_ai_params": { # params for dheera_ai completion/embedding call 
        "model": "replicate/mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70", 
        "api_key": "replicate_api_key",
    }
}, {
	"model_name": "mistral-7b-instruct",
	"dheera_ai_params": { # params for dheera_ai completion/embedding call 
        "model": "together_ai/mistralai/Mistral-7B-Instruct-v0.1", 
        "api_key": "togetherai_api_key",
    }
}, {
	"model_name": "mistral-7b-instruct",
	"dheera_ai_params": { # params for dheera_ai completion/embedding call 
        "model": "together_ai/mistralai/Mistral-7B-Instruct-v0.1", 
        "api_key": "togetherai_api_key",
    }
}, {
	"model_name": "mistral-7b-instruct",
	"dheera_ai_params": { # params for dheera_ai completion/embedding call 
        "model": "perplexity/mistral-7b-instruct", 
		"api_key": "perplexity_api_key"
    }
}, {
	"model_name": "mistral-7b-instruct",
	"dheera_ai_params": {
		"model": "deepinfra/mistralai/Mistral-7B-Instruct-v0.1",
		"api_key": "deepinfra_api_key"
	}
}]

## Dheera AI completion call ## returns first response 
response = completion(model="mistral-7b-instruct", messages=messages, model_list=model_list)

print(response)
```