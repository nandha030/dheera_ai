# DheeraLLM Vector Store & Configuration Guide

## 📚 Table of Contents
1. [Vector Store Setup](#vector-store-setup)
2. [Model Tags & Metadata](#model-tags--metadata)
3. [Advanced Configuration Parameters](#advanced-configuration-parameters)
4. [Complete Examples](#complete-examples)

---

## 🗄️ Vector Store Setup

### Supported Vector Stores
DheeraLLM supports multiple vector store backends:
- **OpenAI Vector Stores** (native integration)
- **Milvus** (self-hosted or cloud)
- **Qdrant** (for semantic caching)
- **Redis** (for semantic caching)

### 1. OpenAI Vector Stores

#### Creating a Vector Store via API

```bash
curl -X POST http://localhost:4000/v1/vector_stores \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-documents",
    "file_ids": ["file-abc123"],
    "expires_after": {
      "anchor": "last_active_at",
      "days": 7
    },
    "metadata": {
      "project": "dheerallm",
      "environment": "production"
    }
  }'
```

#### Vector Store Configuration in YAML

```yaml
# Add to proxy_server_config.yaml
model_list:
  - model_name: gpt-4-with-vector-store
    dheera_ai_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: chat
      supports_vector_store: true
      # Vector store can be attached per-request
```

### 2. Milvus Vector Store

#### Installation
```bash
# Install Milvus dependencies
pip install pymilvus

# Or add to your requirements
pip install dheera_ai[milvus]
```

#### Configuration
```yaml
model_list:
  - model_name: milvus-embeddings
    dheera_ai_params:
      model: openai/text-embedding-ada-002
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: embedding
      vector_store_backend: milvus
      vector_store_config:
        host: localhost
        port: 19530
        collection_name: dheerallm_embeddings
```

### 3. Semantic Caching with Vector Stores

#### Redis Semantic Cache
```yaml
dheera_ai_settings:
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    # Semantic cache configuration
    similarity_threshold: 0.8
    supported_call_types: ["acompletion", "completion", "embedding"]
```

#### Qdrant Semantic Cache
```yaml
dheera_ai_settings:
  cache: true
  cache_params:
    type: qdrant
    collection_name: dheera_ai_semantic_cache
    qdrant_api_base: http://localhost:6333
    similarity_threshold: 0.9
```

---

## 🏷️ Model Tags & Metadata

### Using Tags for Organization

Tags help you organize and filter models in your deployment.

```yaml
model_list:
  - model_name: gpt-4o-production
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      id: "prod-gpt4"
      tags: ["production", "premium", "gpt4"]
      metadata:
        department: "engineering"
        cost_center: "ml-ops"
        tier: "premium"

  - model_name: ollama-dev
    dheera_ai_params:
      model: ollama/phi3:mini
      api_base: http://localhost:11434
    model_info:
      id: "dev-ollama-phi3"
      tags: ["development", "local", "free"]
      metadata:
        environment: "dev"
        gpu_required: false

  - model_name: claude-analysis
    dheera_ai_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      id: "prod-claude"
      tags: ["production", "analysis", "premium"]
      metadata:
        use_case: "code_analysis"
        max_tokens: 200000
```

### Filtering by Tags via API

```bash
# Get models with specific tags
curl http://localhost:4000/v1/models?tags=production,premium \
  -H "Authorization: Bearer sk-1234"
```

---

## ⚙️ Advanced Configuration Parameters

### Complete Model Configuration Options

```yaml
model_list:
  - model_name: advanced-gpt4-config
    dheera_ai_params:
      # Core parameters
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1  # Optional custom endpoint

      # Rate limiting
      rpm: 500              # Requests per minute
      tpm: 90000           # Tokens per minute
      max_parallel_requests: 100

      # Timeout configuration
      timeout: 600         # Request timeout in seconds
      stream_timeout: 60   # Streaming timeout

      # Retry configuration
      num_retries: 3
      retry_after: 5       # Seconds to wait between retries

      # Cost tracking
      input_cost_per_token: 0.00003
      output_cost_per_token: 0.00006

    model_info:
      # Model identification
      id: "advanced-gpt4"
      base_model: "gpt-4o"
      mode: "chat"         # Options: chat, completion, embedding, image, audio

      # Organization
      tags: ["production", "premium", "analysis"]
      metadata:
        department: "research"
        approved_by: "ml-team"
        cost_center: "ai-ops"
        region: "us-east-1"

      # Performance & behavior
      supports_function_calling: true
      supports_vision: true
      supports_parallel_function_calling: true

      # Health checks
      health_check_timeout: 30

      # Context window
      max_tokens: 128000
      max_input_tokens: 120000
      max_output_tokens: 8000
```

### Router Settings with Tags

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2

  # Model group aliases for easier routing
  model_group_alias:
    "premium-models": ["gpt-4o-production", "claude-analysis"]
    "dev-models": ["ollama-dev", "phi3-local"]
    "embedding-models": ["text-embedding-ada-002", "milvus-embeddings"]

  # Enable pre-call health checks
  enable_pre_call_checks: true

  # Fallback configuration
  allowed_fails: 3
  cooldown_time: 60  # seconds

  # Redis for distributed routing (optional)
  redis_host: os.environ/REDIS_HOST
  redis_password: os.environ/REDIS_PASSWORD
  redis_port: os.environ/REDIS_PORT
```

### Team-Based Configuration with Tags

```yaml
dheera_ai_settings:
  default_team_settings:
    - team_id: team-engineering
      tags: ["production", "premium"]  # Team has access to these tagged models
      success_callback: ["langfuse"]
      langfuse_public_key: os.environ/LANGFUSE_ENG_PUBLIC
      langfuse_secret: os.environ/LANGFUSE_ENG_SECRET
      max_budget: 10000
      budget_duration: 30d

    - team_id: team-research
      tags: ["research", "experimental", "premium"]
      success_callback: ["langfuse"]
      langfuse_public_key: os.environ/LANGFUSE_RESEARCH_PUBLIC
      langfuse_secret: os.environ/LANGFUSE_RESEARCH_SECRET
      max_budget: 5000
      budget_duration: 30d

    - team_id: team-development
      tags: ["development", "local", "free"]
      max_budget: 100
      budget_duration: 30d
```

---

## 📋 Complete Examples

### Example 1: Production Setup with Vector Store & Embeddings

```yaml
model_list:
  # Chat models
  - model_name: gpt-4o-prod
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
      timeout: 600
    model_info:
      id: "gpt4-prod-001"
      tags: ["production", "chat", "premium"]
      metadata:
        tier: "premium"
        use_case: "customer_support"

  # Embedding model
  - model_name: embeddings-prod
    dheera_ai_params:
      model: openai/text-embedding-3-large
      api_key: os.environ/OPENAI_API_KEY
      rpm: 3000
    model_info:
      mode: embedding
      id: "embed-prod-001"
      tags: ["production", "embedding"]
      metadata:
        dimensions: 3072
        use_case: "semantic_search"

  # Local development model
  - model_name: ollama-local
    dheera_ai_params:
      model: ollama/phi3:mini
      api_base: http://localhost:11434
    model_info:
      id: "ollama-dev-001"
      tags: ["development", "local", "free"]
      metadata:
        environment: "dev"

dheera_ai_settings:
  # Enable semantic caching
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    similarity_threshold: 0.85

  # Callbacks for monitoring
  success_callback: ["langfuse", "prometheus"]

  # Rate limiting
  num_retries: 5
  request_timeout: 600

  # Context window fallbacks
  context_window_fallbacks:
    - gpt-4o-prod: [gpt-3.5-turbo]

router_settings:
  routing_strategy: usage-based-routing-v2
  enable_pre_call_checks: true
  model_group_alias:
    "chat-models": "gpt-4o-prod"
    "embed-models": "embeddings-prod"
    "dev-models": "ollama-local"

general_settings:
  master_key: sk-1234
  store_model_in_db: true
  database_url: "postgresql://nandhavignesh@localhost:5432/dheera_ai"
```

### Example 2: Multi-Provider Setup with Tags

```yaml
model_list:
  # OpenAI models
  - model_name: gpt-4o
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      tags: ["openai", "premium", "chat"]

  # Anthropic models
  - model_name: claude-sonnet
    dheera_ai_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      tags: ["anthropic", "premium", "chat", "analysis"]

  # Google models
  - model_name: gemini-flash
    dheera_ai_params:
      model: gemini/gemini-1.5-flash
      api_key: os.environ/GOOGLE_API_KEY
    model_info:
      tags: ["google", "fast", "chat"]

  # Local models
  - model_name: "ollama/*"
    dheera_ai_params:
      model: "ollama/*"
      api_base: http://localhost:11434
    model_info:
      tags: ["local", "free", "development"]

router_settings:
  routing_strategy: usage-based-routing-v2
  model_group_alias:
    "premium-ai": ["gpt-4o", "claude-sonnet"]
    "fast-ai": ["gemini-flash"]
    "local-ai": ["ollama/*"]
```

---

## 🚀 Quick Start Commands

### 1. Start DheeraLLM Server
```bash
cd /Users/nandhavignesh/Freedom_AI/Dheera_AI/dheera_ai
source /Users/nandhavignesh/Freedom_AI/Dheera_AI/venv/bin/activate
export DATABASE_URL="postgresql://nandhavignesh@localhost:5432/dheera_ai"
dheera-ai --config proxy_server_config.yaml
```

### 2. Create Vector Store
```bash
curl -X POST http://localhost:4000/v1/vector_stores \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-knowledge-base",
    "metadata": {"project": "dheerallm", "version": "v0.3.0"}
  }'
```

### 3. Upload File to Vector Store
```bash
# First, upload a file
curl -X POST http://localhost:4000/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -F "file=@document.pdf" \
  -F "purpose=assistants"

# Then add it to vector store
curl -X POST http://localhost:4000/v1/vector_stores/{vector_store_id}/files \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"file_id": "file-abc123"}'
```

### 4. Query with Vector Store
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is in my documents?"}],
    "tools": [{
      "type": "file_search",
      "file_search": {
        "vector_store_ids": ["vs_abc123"]
      }
    }]
  }'
```

---

## 📖 Additional Resources

- **Official Docs**: https://docs.dheeraai.com
- **Vector Store API**: https://docs.dheeraai.com/docs/vector_stores
- **Model Configuration**: https://docs.dheeraai.com/docs/proxy/configs
- **Caching Guide**: https://docs.dheeraai.com/docs/caching

---

## 🔧 Troubleshooting

### Vector Store Not Working
1. Ensure model supports vector stores (OpenAI models only)
2. Check API key has proper permissions
3. Verify file upload completed successfully

### Tags Not Filtering
1. Tags are case-sensitive
2. Ensure tags are defined in `model_info` section
3. Use comma-separated values for multiple tags

### Semantic Cache Issues
1. Verify Redis/Qdrant is running
2. Check similarity threshold (0.0-1.0)
3. Ensure embedding model is properly configured
