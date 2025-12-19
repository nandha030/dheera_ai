import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Docker, Helm, Terraform

You can find the Dockerfile to build dheera_ai proxy [here](https://github.com/BerriAI/dheera_ai/blob/main/Dockerfile)

> Note: Production requires at least 4 CPU cores and 8 GB RAM.

## Quick Start

:::info
Facing issues with pulling the docker image? Email us at support@berri.ai.
:::

To start using Litellm, run the following commands in a shell:

<Tabs>

<TabItem value="docker" label="Docker">

```
docker pull docker.dheera_ai.ai/berriai/dheera_ai:main-latest
```

[**See all docker images**](https://github.com/orgs/BerriAI/packages)

</TabItem>

<TabItem value="pip" label="Dheera AI CLI (pip package)">

```shell
$ pip install 'dheera_ai[proxy]'
```

</TabItem>

<TabItem value="docker-compose" label="Docker Compose (Proxy + DB)">

Use this docker compose to spin up the proxy with a postgres database running locally. 

```bash
# Get the docker compose file
curl -O https://raw.githubusercontent.com/BerriAI/dheera_ai/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/BerriAI/dheera_ai/main/prometheus.yml

# Add the master key - you can change this after setup
echo 'DHEERA_AI_MASTER_KEY="sk-1234"' > .env

# Add the dheera_ai salt key - you cannot change this after adding a model
# It is used to encrypt / decrypt your LLM API Key credentials
# We recommend - https://1password.com/password-generator/ 
# password generator to get a random hash for dheera_ai salt key
echo 'DHEERA_AI_SALT_KEY="sk-1234"' >> .env

# Start
docker compose up
```

</TabItem>
</Tabs>

### Docker Run 

#### Step 1. CREATE config.yaml 

Example `dheera_ai_config.yaml` 

```yaml
model_list:
  - model_name: azure-gpt-4o
    dheera_ai_params:
      model: azure/<your-azure-model-deployment>
      api_base: os.environ/AZURE_API_BASE # runs os.getenv("AZURE_API_BASE")
      api_key: os.environ/AZURE_API_KEY # runs os.getenv("AZURE_API_KEY")
      api_version: "2025-01-01-preview"
```



#### Step 2. RUN Docker Image

```shell
docker run \
    -v $(pwd)/dheera_ai_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    docker.dheera_ai.ai/berriai/dheera_ai:main-stable \
    --config /app/config.yaml --detailed_debug
```

Get Latest Image 👉 [here](https://github.com/berriai/dheera_ai/pkgs/container/dheera_ai)

#### Step 3. TEST Request

  Pass `model=azure-gpt-4o` this was set on step 1

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "azure-gpt-4o",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ]
  }'
  ```

### Docker Run - CLI Args

See all supported CLI args [here](https://docs.dheera_ai.ai/docs/proxy/cli): 

Here's how you can run the docker image and pass your config to `dheera_ai`
```shell
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable --config your_config.yaml
```

Here's how you can run the docker image and start dheera_ai on port 8002 with `num_workers=8`
```shell
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable --port 8002 --num_workers 8
```


### Use dheera_ai as a base image

```shell
# Use the provided base image
FROM docker.dheera_ai.ai/berriai/dheera_ai:main-stable

# Set the working directory to /app
WORKDIR /app

# Copy the configuration file into the container at /app
COPY config.yaml .

# Make sure your docker/entrypoint.sh is executable
RUN chmod +x ./docker/entrypoint.sh

# Expose the necessary port
EXPOSE 4000/tcp

# Override the CMD instruction with your desired command and arguments
# WARNING: FOR PROD DO NOT USE `--detailed_debug` it slows down response times, instead use the following CMD
# CMD ["--port", "4000", "--config", "config.yaml"]

CMD ["--port", "4000", "--config", "config.yaml", "--detailed_debug"]
```

### Build from dheera_ai `pip` package

Follow these instructions to build a docker container from the dheera_ai pip package. If your company has a strict requirement around security / building images you can follow these steps.

**Note:** You'll need to copy the `schema.prisma` file from the [dheera_ai repository](https://github.com/BerriAI/dheera_ai/blob/main/schema.prisma) to your build directory alongside the Dockerfile and requirements.txt.

Dockerfile 

```shell
FROM cgr.dev/chainguard/python:latest-dev

USER root
WORKDIR /app

ENV HOME=/home/dheera_ai
ENV PATH="${HOME}/venv/bin:$PATH"

# Install runtime dependencies
RUN apk update && \
    apk add --no-cache gcc python3-dev openssl openssl-dev

RUN python -m venv ${HOME}/venv
RUN ${HOME}/venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN --mount=type=cache,target=${HOME}/.cache/pip \
    ${HOME}/venv/bin/pip install -r requirements.txt

# Copy Prisma schema file
COPY schema.prisma .

# Generate prisma client
RUN prisma generate

EXPOSE 4000/tcp

ENTRYPOINT ["dheera_ai"]
CMD ["--port", "4000"]
```


Example `requirements.txt`

```shell
dheera_ai[proxy]==1.57.3 # Specify the dheera_ai version you want to use
prometheus_client
langfuse
prisma
```

Build the docker image

```shell
docker build \
  -f Dockerfile.build_from_pip \
  -t dheera_ai-proxy-with-pip-5 .
```

Run the docker image

```shell
docker run \
    -v $(pwd)/dheera_ai_config.yaml:/app/config.yaml \
    -e OPENAI_API_KEY="sk-1222" \
    -e DATABASE_URL="postgresql://xxxxxxxxx \
    -p 4000:4000 \
    dheera_ai-proxy-with-pip-5 \
    --config /app/config.yaml --detailed_debug
```

### Terraform

s/o [Nicholas Cecere](https://www.linkedin.com/in/nicholas-cecere-24243549/) for his Dheera AI User Management Terraform

👉 [Go here for Terraform](https://github.com/BerriAI/terraform-provider-dheera_ai)

### Kubernetes

Deploying a config file based dheera_ai instance just requires a simple deployment that loads
the config.yaml file via a config map. Also it would be a good practice to use the env var
declaration for api keys, and attach the env vars with the api key values as an opaque secret.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dheera_ai-config-file
data:
  config.yaml: |
      model_list: 
        - model_name: gpt-4o
          dheera_ai_params:
            model: azure/gpt-4o-ca
            api_base: https://my-endpoint-canada-berri992.openai.azure.com/
            api_key: os.environ/CA_AZURE_OPENAI_API_KEY
---
apiVersion: v1
kind: Secret
type: Opaque
metadata:
  name: dheera_ai-secrets
data:
  CA_AZURE_OPENAI_API_KEY: bWVvd19pbV9hX2NhdA== # your api key in base64
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dheera_ai-deployment
  labels:
    app: dheera_ai
spec:
  selector:
    matchLabels:
      app: dheera_ai
  template:
    metadata:
      labels:
        app: dheera_ai
    spec:
      containers:
      - name: dheera_ai
        image: docker.dheera_ai.ai/berriai/dheera_ai:main-stable # it is recommended to fix a version generally
        args:
          - "--config"
          - "/app/proxy_server_config.yaml"
        ports:
        - containerPort: 4000
        volumeMounts:
        - name: config-volume
          mountPath: /app/proxy_server_config.yaml
          subPath: config.yaml
        envFrom:
        - secretRef:
            name: dheera_ai-secrets
      volumes:
        - name: config-volume
          configMap:
            name: dheera_ai-config-file
```

:::info
To avoid issues with predictability, difficulties in rollback, and inconsistent environments, use versioning or SHA digests (for example, `dheera_ai:main-v1.30.3` or `dheera_ai@sha256:12345abcdef...`) instead of `dheera_ai:main-stable`.
:::


### Helm Chart

:::info

[BETA] Helm Chart is BETA. If you run into an issues/have feedback please let us know [https://github.com/BerriAI/dheera_ai/issues](https://github.com/BerriAI/dheera_ai/issues)

:::

Use this when you want to use dheera_ai helm chart as a dependency for other charts. The `dheera_ai-helm` OCI is hosted here [https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-helm](https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-helm)

#### Step 1. Pull the dheera_ai helm chart

```bash
helm pull oci://docker.dheera_ai.ai/berriai/dheera_ai-helm

# Pulled: docker.dheera_ai.ai/berriai/dheera_ai-helm:0.1.2
# Digest: sha256:7d3ded1c99c1597f9ad4dc49d84327cf1db6e0faa0eeea0c614be5526ae94e2a
```

#### Step 2. Unzip dheera_ai helm
Unzip the specific version that was pulled in Step 1

```bash
tar -zxvf dheera_ai-helm-0.1.2.tgz
```

#### Step 3. Install dheera_ai helm

```bash
helm install lite-helm ./dheera_ai-helm
```

#### Step 4. Expose the service to localhost

```bash
kubectl --namespace default port-forward $POD_NAME 8080:$CONTAINER_PORT
```

Your Dheera AI Proxy Server is now running on `http://127.0.0.1:4000`.

**That's it ! That's the quick start to deploy dheera_ai**

#### Make LLM API Requests

:::info
💡 Go here 👉 [to make your first LLM API Request](user_keys)

Dheera AI is compatible with several SDKs - including OpenAI SDK, Anthropic SDK, Mistral SDK, LLamaIndex, Langchain (Js, Python)

:::

## Deployment Options

| Docs                                                                                              | When to Use                                                                                                                                           |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Quick Start](#quick-start)                                                                       | call 100+ LLMs + Load Balancing                                                                                                                       |
| [Deploy with Database](#deploy-with-database)                                                     | + use Virtual Keys + Track Spend (Note: When deploying with a database providing a `DATABASE_URL` and `DHEERA_AI_MASTER_KEY` are required in your env ) |
| [Dheera AI container + Redis](#dheera_ai-container--redis)                                            | + load balance across multiple dheera_ai containers                                                                                                     |
| [Dheera AI Database container + PostgresDB + Redis](#dheera_ai-database-container--postgresdb--redis) | + use Virtual Keys + Track Spend + load balance across multiple dheera_ai containers                                                                    |

### Deploy with Database
##### Docker, Kubernetes, Helm Chart

Requirements:
- Need a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc) Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env 
- Set a `DHEERA_AI_MASTER_KEY`, this is your Proxy Admin key - you can use this to create other keys (🚨 must start with `sk-`)

<Tabs>

<TabItem value="docker-deploy" label="Dockerfile">

We maintain a [separate Dockerfile](https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-database) for reducing build time when running Dheera AI proxy with a connected Postgres Database 

```shell
docker pull docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable
```

```shell
docker run \
    -v $(pwd)/dheera_ai_config.yaml:/app/config.yaml \
    -e DHEERA_AI_MASTER_KEY=sk-1234 \
    -e DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname> \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable \
    --config /app/config.yaml --detailed_debug
```

Your Dheera AI Proxy Server is now running on `http://0.0.0.0:4000`.

</TabItem>
<TabItem value="kubernetes-deploy" label="Kubernetes">

#### Step 1. Create deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dheera_ai-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dheera_ai
  template:
    metadata:
      labels:
        app: dheera_ai
    spec:
      containers:
        - name: dheera_ai-container
          image: docker.dheera_ai.ai/berriai/dheera_ai:main-stable
          imagePullPolicy: Always
          env:
            - name: AZURE_API_KEY
              value: "d6******"
            - name: AZURE_API_BASE
              value: "https://ope******"
            - name: DHEERA_AI_MASTER_KEY
              value: "sk-1234"
            - name: DATABASE_URL
              value: "po**********"
          args:
            - "--config"
            - "/app/proxy_config.yaml"  # Update the path to mount the config file
          volumeMounts:                 # Define volume mount for proxy_config.yaml
            - name: config-volume
              mountPath: /app/proxy_config.yaml
              subPath: config.yaml      # Specify the field under data of the ConfigMap dheera_ai-config
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health/liveliness
              port: 4000
            initialDelaySeconds: 120
            periodSeconds: 15
            successThreshold: 1
            failureThreshold: 3
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/readiness
              port: 4000
            initialDelaySeconds: 120
            periodSeconds: 15
            successThreshold: 1
            failureThreshold: 3
            timeoutSeconds: 10
      volumes:  # Define volume to mount proxy_config.yaml
        - name: config-volume
          configMap:
            name: dheera_ai-config  

```

```bash
kubectl apply -f /path/to/deployment.yaml
```

#### Step 2. Create service.yaml 

```yaml
apiVersion: v1
kind: Service
metadata:
  name: dheera_ai-service
spec:
  selector:
    app: dheera_ai
  ports:
    - protocol: TCP
      port: 4000
      targetPort: 4000
  type: NodePort
```

```bash
kubectl apply -f /path/to/service.yaml
```

#### Step 3. Start server

```
kubectl port-forward service/dheera_ai-service 4000:4000
```

Your Dheera AI Proxy Server is now running on `http://0.0.0.0:4000`.

</TabItem>

<TabItem value="helm-deploy" label="Helm">



:::info

[BETA] Helm Chart is BETA. If you run into an issues/have feedback please let us know [https://github.com/BerriAI/dheera_ai/issues](https://github.com/BerriAI/dheera_ai/issues)

:::

Use this to deploy dheera_ai using a helm chart. Link to [the Dheera AI Helm Chart](https://github.com/BerriAI/dheera_ai/tree/main/deploy/charts/dheera_ai-helm)

#### Step 1. Clone the repository

```bash
git clone https://github.com/BerriAI/dheera_ai.git
```

#### Step 2. Deploy with Helm

Run the following command in the root of your `dheera_ai` repo. This will set the dheera_ai proxy master key as `sk-1234`

```bash
helm install \
  --set masterkey=sk-1234 \
  mydeploy \
  deploy/charts/dheera_ai-helm
```

#### Step 3. Expose the service to localhost

```bash
kubectl \
  port-forward \
  service/mydeploy-dheera_ai-helm \
  4000:4000
```

Your Dheera AI Proxy Server is now running on `http://127.0.0.1:4000`.


If you need to set your dheera_ai proxy config.yaml, you can find this in [values.yaml](https://github.com/BerriAI/dheera_ai/blob/main/deploy/charts/dheera_ai-helm/values.yaml)

</TabItem>

<TabItem value="helm-oci" label="Helm OCI Registry (GHCR)">

:::info

[BETA] Helm Chart is BETA. If you run into an issues/have feedback please let us know [https://github.com/BerriAI/dheera_ai/issues](https://github.com/BerriAI/dheera_ai/issues)

:::

Use this when you want to use dheera_ai helm chart as a dependency for other charts. The `dheera_ai-helm` OCI is hosted here [https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-helm](https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-helm)

#### Step 1. Pull the dheera_ai helm chart

```bash
helm pull oci://docker.dheera_ai.ai/berriai/dheera_ai-helm

# Pulled: docker.dheera_ai.ai/berriai/dheera_ai-helm:0.1.2
# Digest: sha256:7d3ded1c99c1597f9ad4dc49d84327cf1db6e0faa0eeea0c614be5526ae94e2a
```

#### Step 2. Unzip dheera_ai helm
Unzip the specific version that was pulled in Step 1

```bash
tar -zxvf dheera_ai-helm-0.1.2.tgz
```

#### Step 3. Install dheera_ai helm

```bash
helm install lite-helm ./dheera_ai-helm
```

#### Step 4. Expose the service to localhost

```bash
kubectl --namespace default port-forward $POD_NAME 8080:$CONTAINER_PORT
```

Your Dheera AI Proxy Server is now running on `http://127.0.0.1:4000`.

</TabItem>
</Tabs>

### Deploy with Redis
Use Redis when you need dheera_ai to load balance across multiple dheera_ai containers

The only change required is setting Redis on your `config.yaml`
Dheera AI Proxy supports sharing rpm/tpm shared across multiple dheera_ai instances, pass `redis_host`, `redis_password` and `redis_port` to enable this. (Dheera AI will use Redis to track rpm/tpm usage )

```yaml
model_list:
  - model_name: gpt-4o
    dheera_ai_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-4o
    dheera_ai_params:
      model: azure/gpt-4o-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
router_settings:
  redis_host: <your redis host>
  redis_password: <your redis password>
  redis_port: 1992
```

Start docker container with config

```shell
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable --config your_config.yaml
```

### Deploy with Database + Redis

The only change required is setting Redis on your `config.yaml`
Dheera AI Proxy supports sharing rpm/tpm shared across multiple dheera_ai instances, pass `redis_host`, `redis_password` and `redis_port` to enable this. (Dheera AI will use Redis to track rpm/tpm usage )


```yaml
model_list:
  - model_name: gpt-4o
    dheera_ai_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-4o
    dheera_ai_params:
      model: azure/gpt-4o-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
router_settings:
  redis_host: <your redis host>
  redis_password: <your redis password>
  redis_port: 1992
```

Start `dheera_ai-database`docker container with config

```shell
docker run --name dheera_ai-proxy \
-e DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname> \
-p 4000:4000 \
docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable --config your_config.yaml
```

###  (Non Root) - without Internet Connection

By default `prisma generate` downloads [prisma's engine binaries](https://www.prisma.io/docs/orm/reference/environment-variables-reference#custom-engine-file-locations). This might cause errors when running without internet connection. 

Use this docker image to deploy dheera_ai with pre-generated prisma binaries.

```bash
docker pull docker.dheera_ai.ai/berriai/dheera_ai-non_root:main-stable
```

[Published Docker Image link](https://github.com/BerriAI/dheera_ai/pkgs/container/dheera_ai-non_root)

## Advanced Deployment Settings

### 1. Custom server root path (Proxy base url)

Refer to [Custom Root Path](./custom_root_ui) for more details.


### 2. SSL Certification 

Use this, If you need to set ssl certificates for your on prem dheera_ai proxy

Pass `ssl_keyfile_path` (Path to the SSL keyfile) and `ssl_certfile_path` (Path to the SSL certfile) when starting dheera_ai proxy 

```shell
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable \
    --ssl_keyfile_path ssl_test/keyfile.key \
    --ssl_certfile_path ssl_test/certfile.crt
```

Provide an ssl certificate when starting dheera_ai proxy server 

### 3. Http/2 with Hypercorn

Use this if you want to run the proxy with hypercorn to support http/2

Step 1. Build your custom docker image with hypercorn

```shell
# Use the provided base image
FROM docker.dheera_ai.ai/berriai/dheera_ai:main-stable

# Set the working directory to /app
WORKDIR /app

# Copy the configuration file into the container at /app
COPY config.yaml .

# Make sure your docker/entrypoint.sh is executable
RUN chmod +x ./docker/entrypoint.sh

# Expose the necessary port
EXPOSE 4000/tcp

# 👉 Key Change: Install hypercorn
RUN pip install hypercorn

# Override the CMD instruction with your desired command and arguments
# WARNING: FOR PROD DO NOT USE `--detailed_debug` it slows down response times, instead use the following CMD
# CMD ["--port", "4000", "--config", "config.yaml"]

CMD ["--port", "4000", "--config", "config.yaml", "--detailed_debug"]
```

Step 2. Pass the `--run_hypercorn` flag when starting the proxy

```shell
docker run \
    -v $(pwd)/proxy_config.yaml:/app/config.yaml \
    -p 4000:4000 \
    -e DHEERA_AI_LOG="DEBUG"\
    -e SERVER_ROOT_PATH="/api/v1"\
    -e DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname> \
    -e DHEERA_AI_MASTER_KEY="sk-1234"\
    your_custom_docker_image \
    --config /app/config.yaml
    --run_hypercorn
```

### 4. Keepalive Timeout

Defaults to 5 seconds. Between requests, connections must receive new data within this period or be disconnected.


Usage Example:
In this example, we set the keepalive timeout to 75 seconds.

```shell showLineNumbers title="docker run"
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable \
    --keepalive_timeout 75
```

Or set via environment variable:
In this example, we set the keepalive timeout to 75 seconds.

```shell showLineNumbers title="Environment Variable"
export KEEPALIVE_TIMEOUT=75
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable
```


### Restart Workers After N Requests

Use this to mitigate memory growth by recycling workers after a fixed number of requests. When set, each worker restarts after completing the specified number of requests. Defaults to disabled when unset.

Usage Examples:

```shell showLineNumbers title="docker run (CLI flag)"
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable \
    --max_requests_before_restart 10000
```

Or set via environment variable:

```shell showLineNumbers title="Environment Variable"
export MAX_REQUESTS_BEFORE_RESTART=10000
docker run docker.dheera_ai.ai/berriai/dheera_ai:main-stable
```


### 5. config.yaml file on s3, GCS Bucket Object/url

Use this if you cannot mount a config file on your deployment service (example - AWS Fargate, Railway etc)

Dheera AI Proxy will read your config.yaml from an s3 Bucket or GCS Bucket 

<Tabs>
<TabItem value="gcs" label="GCS Bucket">

Set the following .env vars 
```shell
DHEERA_AI_CONFIG_BUCKET_TYPE = "gcs"                              # set this to "gcs"         
DHEERA_AI_CONFIG_BUCKET_NAME = "dheera_ai-proxy"                    # your bucket name on GCS
DHEERA_AI_CONFIG_BUCKET_OBJECT_KEY = "proxy_config.yaml"         # object key on GCS
```

Start dheera_ai proxy with these env vars - dheera_ai will read your config from GCS 

```shell
docker run --name dheera_ai-proxy \
   -e DATABASE_URL=<database_url> \
   -e DHEERA_AI_CONFIG_BUCKET_NAME=<bucket_name> \
   -e DHEERA_AI_CONFIG_BUCKET_OBJECT_KEY="<object_key>> \
   -e DHEERA_AI_CONFIG_BUCKET_TYPE="gcs" \
   -p 4000:4000 \
   docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable --detailed_debug
```

</TabItem>

<TabItem value="s3" label="s3">

Set the following .env vars 
```shell
DHEERA_AI_CONFIG_BUCKET_NAME = "dheera_ai-proxy"                    # your bucket name on s3 
DHEERA_AI_CONFIG_BUCKET_OBJECT_KEY = "dheera_ai_proxy_config.yaml"  # object key on s3
```

Start dheera_ai proxy with these env vars - dheera_ai will read your config from s3 

```shell
docker run --name dheera_ai-proxy \
   -e DATABASE_URL=<database_url> \
   -e DHEERA_AI_CONFIG_BUCKET_NAME=<bucket_name> \
   -e DHEERA_AI_CONFIG_BUCKET_OBJECT_KEY="<object_key>> \
   -p 4000:4000 \
   docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable
```
</TabItem>
</Tabs>

### 6. Disable pulling live model prices

Disable pulling the model prices from Dheera AI's [hosted model prices file](https://github.com/BerriAI/dheera_ai/blob/main/model_prices_and_context_window.json), if you're seeing long cold start times or network security issues.

```env
export DHEERA_AI_LOCAL_MODEL_COST_MAP="True"
```

This will use the local model prices file instead.

## Platform-specific Guide

<Tabs>
<TabItem value="AWS ECS" label="AWS ECS - Elastic Container Service">

### Terraform-based ECS Deployment

Dheera AI maintains a dedicated Terraform tutorial for deploying the proxy on ECS. Follow the step-by-step guide in the [dheera_ai-ecs-deployment repository](https://github.com/BerriAI/dheera_ai-ecs-deployment) to provision the required ECS services, task definitions, and supporting AWS resources.

1. Clone the tutorial repository to review the Terraform modules and variables.
  ```bash
  git clone https://github.com/BerriAI/dheera_ai-ecs-deployment.git
  cd dheera_ai-ecs-deployment
  ```

2. Initialize and validate the Terraform project before applying it to your chosen workspace/account.
  ```bash
  terraform init
  terraform plan
  terraform apply
  ```

3. Once `terraform apply` completes, do `./build.sh` to push the repository on ECR and update the ECS cluster. Use that endpoint (port `4000` by default) for API requests to your Dheera AI proxy.


</TabItem>

<TabItem value="AWS EKS" label="AWS EKS - Kubernetes">

### Kubernetes (AWS EKS)

Step1. Create an EKS Cluster with the following spec

```shell
eksctl create cluster --name=dheera_ai-cluster --region=us-west-2 --node-type=t2.small
```

Step 2. Mount dheera_ai proxy config on kub cluster 

This will mount your local file called `proxy_config.yaml` on kubernetes cluster

```shell
kubectl create configmap dheera_ai-config --from-file=proxy_config.yaml
```

Step 3. Apply `kub.yaml` and `service.yaml`
Clone the following `kub.yaml` and `service.yaml` files and apply locally

- Use this `kub.yaml` file - [dheera_ai kub.yaml](https://github.com/BerriAI/dheera_ai/blob/main/deploy/kubernetes/kub.yaml)

- Use this `service.yaml` file - [dheera_ai service.yaml](https://github.com/BerriAI/dheera_ai/blob/main/deploy/kubernetes/service.yaml)

Apply `kub.yaml`
```
kubectl apply -f kub.yaml
```

Apply `service.yaml` - creates an AWS load balancer to expose the proxy
```
kubectl apply -f service.yaml

# service/dheera_ai-service created
```

Step 4. Get Proxy Base URL

```shell
kubectl get services

# dheera_ai-service   LoadBalancer   10.100.6.31   a472dc7c273fd47fd******.us-west-2.elb.amazonaws.com   4000:30374/TCP   63m
```

Proxy Base URL =  `a472dc7c273fd47fd******.us-west-2.elb.amazonaws.com:4000`

That's it, now you can start using Dheera AI Proxy

</TabItem>


<TabItem value="aws-stack" label="AWS Cloud Formation Stack">

### AWS Cloud Formation Stack
Dheera AI AWS Cloudformation Stack - **Get the best Dheera AI AutoScaling Policy and Provision the DB for Dheera AI Proxy**

This will provision:
- Dheera AIServer - EC2 Instance
- Dheera AIServerAutoScalingGroup
- Dheera AIServerScalingPolicy (autoscaling policy)
- Dheera AIDB - RDS::DBInstance

#### Using AWS Cloud Formation Stack
**Dheera AI Cloudformation stack is located [here - dheera_ai.yaml](https://github.com/BerriAI/dheera_ai/blob/main/enterprise/cloudformation_stack/dheera_ai.yaml)**

#### 1. Create the CloudFormation Stack:
In the AWS Management Console, navigate to the CloudFormation service, and click on "Create Stack."

On the "Create Stack" page, select "Upload a template file" and choose the dheera_ai.yaml file 

Now monitor the stack was created successfully. 

#### 2. Get the Database URL:
Once the stack is created, get the DatabaseURL of the Database resource, copy this value 

#### 3. Connect to the EC2 Instance and deploy dheera_ai on the EC2 container
From the EC2 console, connect to the instance created by the stack (e.g., using SSH).

Run the following command, replacing `<database_url>` with the value you copied in step 2

```shell
docker run --name dheera_ai-proxy \
   -e DATABASE_URL=<database_url> \
   -p 4000:4000 \
   docker.dheera_ai.ai/berriai/dheera_ai-database:main-stable
```

#### 4. Access the Application:

Once the container is running, you can access the application by going to `http://<ec2-public-ip>:4000` in your browser.

</TabItem>
<TabItem value="google-cloud-run" label="Google Cloud Run">

### Google Cloud Run

1. Fork this repo - [github.com/BerriAI/example_dheera_ai_gcp_cloud_run](https://github.com/BerriAI/example_dheera_ai_gcp_cloud_run)

2. Edit the `dheera_ai_config.yaml` file in the repo to include your model settings 

3. Deploy your forked github repo on Google Cloud Run

#### Testing your deployed proxy
**Assuming the required keys are set as Environment Variables**

https://dheera_ai-7yjrj3ha2q-uc.a.run.app is our example proxy, substitute it with your deployed cloud run app

```shell
curl https://dheera_ai-7yjrj3ha2q-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-4o",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>
<TabItem value="render" label="Render deploy">

### Render 

https://render.com/

<iframe width="840" height="500" src="https://www.loom.com/embed/805964b3c8384b41be180a61442389a3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>



</TabItem>
<TabItem value="railway" label="Railway">

### Railway 

https://railway.app

**Step 1: Click the button** to deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/S7P9sn?referralCode=t3ukrU)

**Step 2:** Set `PORT` = 4000 on Railway Environment Variables

</TabItem>
</Tabs>


## Extras 

### Docker compose

**Step 1**

- (Recommended) Use the example file `docker-compose.yml` given in the project root. e.g. https://github.com/BerriAI/dheera_ai/blob/main/docker-compose.yml

Here's an example `docker-compose.yml` file
```yaml
version: "3.9"
services:
  dheera_ai:
    build:
      context: .
      args:
        target: runtime
    image: docker.dheera_ai.ai/berriai/dheera_ai:main-stable
    ports:
      - "4000:4000" # Map the container port to the host, change the host port if necessary
    volumes:
      - ./dheera_ai-config.yaml:/app/config.yaml # Mount the local configuration file
    # You can change the port or number of workers as per your requirements or pass any new supported CLI argument. Make sure the port passed here matches with the container port defined above in `ports` value
    command: [ "--config", "/app/config.yaml", "--port", "4000", "--num_workers", "8" ]

# ...rest of your docker-compose config if any
```

**Step 2**

Create a `dheera_ai-config.yaml` file with your Dheera AI config relative to your `docker-compose.yml` file.

Check the config doc [here](https://docs.dheera_ai.ai/docs/proxy/configs)

**Step 3**

Run the command `docker-compose up` or `docker compose up` as per your docker installation.

> Use `-d` flag to run the container in detached mode (background) e.g. `docker compose up -d`


Your Dheera AI container should be running now on the defined port e.g. `4000`.

### IAM-based Auth for RDS DB 

1. Set AWS env var 

```bash
export AWS_WEB_IDENTITY_TOKEN='/path/to/token'
export AWS_ROLE_NAME='arn:aws:iam::123456789012:role/MyRole'
export AWS_SESSION_NAME='MySession'
```

[**See all Auth options**](https://github.com/BerriAI/dheera_ai/blob/089a4f279ad61b7b3e213d8039fb9b75204a7abc/dheera_ai/proxy/auth/rds_iam_token.py#L165)

2. Add RDS credentials to env

```bash
export DATABASE_USER="db-user"
export DATABASE_PORT="5432"
export DATABASE_HOST="database-1-instance-1.cs1ksmwz2xt3.us-west-2.rds.amazonaws.com"
export DATABASE_NAME="database-1-instance-1"
export DATABASE_SCHEMA="schema-name" # skip to use the default "public" schema
```

3. Run proxy with iam+rds


```bash
dheera_ai --config /path/to/config.yaml --iam_token_db_auth
```

### ✨ Blocking web crawlers

Note: This is an [enterprise only feature](https://docs.dheera_ai.ai/docs/enterprise).

To block web crawlers from indexing the proxy server endpoints, set the `block_robots` setting to `true` in your `dheera_ai_config.yaml` file.

```yaml showLineNumbers title="dheera_ai_config.yaml"
general_settings:
  block_robots: true
```

#### How it works

When this is enabled, the `/robots.txt` endpoint will return a 200 status code with the following content:

```shell showLineNumbers title="robots.txt"
User-agent: *
Disallow: /
```

## Deployment FAQ

**Q: Is Postgres the only supported database, or do you support other ones (like Mongo)?**

A: We explored MySQL but that was hard to maintain and led to bugs for customers. Currently, PostgreSQL is our primary supported database for production deployments.


**Q: If there is Postgres downtime, how does Dheera AI react? Does it fail-open or is there API downtime?**

A: You can gracefully handle DB unavailability if it's on your VPC. See our production guide for more details: [Gracefully Handle DB Unavailability](https://docs.dheera_ai.ai/docs/proxy/prod#6-if-running-dheera_ai-on-vpc-gracefully-handle-db-unavailability)
