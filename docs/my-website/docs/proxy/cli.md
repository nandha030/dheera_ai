# CLI Arguments
Cli arguments,  --host, --port, --num_workers

## --host
   - **Default:** `'0.0.0.0'`
   - The host for the server to listen on.
   - **Usage:** 
     ```shell
     dheera_ai --host 127.0.0.1
     ```
   - **Usage - set Environment Variable:** `HOST`
    ```shell
    export HOST=127.0.0.1
    dheera_ai
    ```

## --port
   - **Default:** `4000`
   - The port to bind the server to.
   - **Usage:** 
     ```shell
     dheera_ai --port 8080
     ```
  - **Usage - set Environment Variable:** `PORT`
    ```shell
    export PORT=8080
    dheera_ai
    ```

## --num_workers
   - **Default:** `1`
   - The number of uvicorn workers to spin up.
   - **Usage:** 
     ```shell
     dheera_ai --num_workers 4
     ```
  - **Usage - set Environment Variable:** `NUM_WORKERS`
    ```shell
    export NUM_WORKERS=4
    dheera_ai
    ```

## --api_base
   - **Default:** `None`
   - The API base for the model dheera_ai should call.
   - **Usage:** 
     ```shell
     dheera_ai --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
     ```

## --api_version
   - **Default:** `None`
   - For Azure services, specify the API version.
   - **Usage:** 
     ```shell
     dheera_ai --model azure/gpt-deployment --api_version 2023-08-01 --api_base https://<your api base>"
     ```

## --model or -m
   - **Default:** `None`
   - The model name to pass to Litellm.
   - **Usage:** 
     ```shell
     dheera_ai --model gpt-3.5-turbo
     ```

## --test
   - **Type:** `bool` (Flag)
   - Proxy chat completions URL to make a test request.
   - **Usage:** 
     ```shell
     dheera_ai --test
     ```

## --health
   - **Type:** `bool` (Flag)
   - Runs a health check on all models in config.yaml
   - **Usage:** 
     ```shell
     dheera_ai --health
     ```

## --alias
   - **Default:** `None`
   - An alias for the model, for user-friendly reference.
   - **Usage:** 
     ```shell
     dheera_ai --alias my-gpt-model
     ```

## --debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable debugging mode for the input.
   - **Usage:** 
     ```shell
     dheera_ai --debug
     ```
  - **Usage - set Environment Variable:** `DEBUG`
    ```shell
    export DEBUG=True
    dheera_ai
    ```

## --detailed_debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable debugging mode for the input.
   - **Usage:** 
     ```shell
     dheera_ai --detailed_debug
     ```
  - **Usage - set Environment Variable:** `DETAILED_DEBUG`
    ```shell
    export DETAILED_DEBUG=True
    dheera_ai
    ```

#### --temperature
   - **Default:** `None`
   - **Type:** `float`
   - Set the temperature for the model.
   - **Usage:** 
     ```shell
     dheera_ai --temperature 0.7
     ```

## --max_tokens
   - **Default:** `None`
   - **Type:** `int`
   - Set the maximum number of tokens for the model output.
   - **Usage:** 
     ```shell
     dheera_ai --max_tokens 50
     ```

## --request_timeout
   - **Default:** `6000`
   - **Type:** `int`
   - Set the timeout in seconds for completion calls.
   - **Usage:** 
     ```shell
     dheera_ai --request_timeout 300
     ```

## --drop_params
   - **Type:** `bool` (Flag)
   - Drop any unmapped params.
   - **Usage:** 
     ```shell
     dheera_ai --drop_params
     ```

## --add_function_to_prompt
   - **Type:** `bool` (Flag)
   - If a function passed but unsupported, pass it as a part of the prompt.
   - **Usage:** 
     ```shell
     dheera_ai --add_function_to_prompt
     ```

## --config
   - Configure Litellm by providing a configuration file path.
   - **Usage:** 
     ```shell
     dheera_ai --config path/to/config.yaml
     ```

## --telemetry
   - **Default:** `True`
   - **Type:** `bool`
   - Help track usage of this feature.
   - **Usage:** 
     ```shell
     dheera_ai --telemetry False
     ```


## --log_config
   - **Default:** `None`
   - **Type:** `str`
   - Specify a log configuration file for uvicorn.
   - **Usage:** 
     ```shell
     dheera_ai --log_config path/to/log_config.conf
     ```

## --skip_server_startup
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Skip starting the server after setup (useful for DB migrations only).
   - **Usage:** 
     ```shell
     dheera_ai --skip_server_startup
     ```