# DheeraLLM v0.3.0

```
   ██████╗ ██╗  ██╗███████╗███████╗██████╗  █████╗ ██╗     ██╗     ███╗   ███╗
   ██╔══██╗██║  ██║██╔════╝██╔════╝██╔══██╗██╔══██╗██║     ██║     ████╗ ████║
   ██║  ██║███████║█████╗  █████╗  ██████╔╝███████║██║     ██║     ██╔████╔██║
   ██║  ██║██╔══██║██╔══╝  ██╔══╝  ██╔══██╗██╔══██║██║     ██║     ██║╚██╔╝██║
   ██████╔╝██║  ██║███████╗███████╗██║  ██║██║  ██║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝
```

**DheeraLLM** is a unified LLM proxy server that provides a consistent API interface for 100+ LLM providers including OpenAI, Anthropic, Google, Azure, Ollama, and more.

[![GitHub](https://img.shields.io/badge/GitHub-nandha030%2Fdheera__ai-blue?logo=github)](https://github.com/nandha030/dheera_ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)](https://www.postgresql.org/)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Node.js v20 (for UI development)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nandha030/dheera_ai.git
cd Dheera_AI
```

2. **Set up virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e dheera_ai
```

3. **Set up database:**
```bash
# Create PostgreSQL database
createdb dheera_ai

# Set environment variable
export DATABASE_URL="postgresql://nandhavignesh@localhost:5432/dheera_ai"

# Run migrations
cd dheera_ai
prisma db push --schema=./dheera_ai/proxy/schema.prisma
```

4. **Start the server:**
```bash
./dheera-start
```

The server will be available at:
- **UI**: http://localhost:4000/ui
- **API**: http://localhost:4000
- **Docs**: http://localhost:4000/docs

### Default Credentials
- **Email**: admin@dheeraai.com
- **Password**: admin123

---

## 📋 Features

### ✨ Core Features
- **100+ LLM Providers**: OpenAI, Anthropic, Google, Azure, AWS Bedrock, Ollama, and more
- **Unified API**: Single API interface for all providers
- **Local LLM Support**: Run models locally with Ollama integration
- **Vector Stores**: Built-in support for OpenAI Vector Stores, Milvus, Qdrant
- **Semantic Caching**: Redis and Qdrant semantic caching for faster responses
- **Load Balancing**: Multiple routing strategies including usage-based routing
- **Rate Limiting**: Per-model and per-team rate limits
- **Cost Tracking**: Track usage and costs across all providers
- **Team Management**: Multi-tenant support with team-based access control

### 🎯 Advanced Features
- **Model Fallbacks**: Automatic fallback to alternative models
- **Health Checks**: Pre-call health checks for high availability
- **Streaming Support**: Real-time streaming for all supported models
- **Custom Callbacks**: Integration with Langfuse, Prometheus, and custom webhooks
- **Tags & Metadata**: Organize models with tags and custom metadata
- **API Key Management**: Create and manage API keys with usage limits

---

## 🛠️ Server Management

### Start Server
```bash
./dheera-start
```

### Stop Server
```bash
./dheera-stop
```

### Manual Start
```bash
cd /Users/nandhavignesh/Freedom_AI/Dheera_AI/dheera_ai
source /Users/nandhavignesh/Freedom_AI/Dheera_AI/venv/bin/activate
export DATABASE_URL="postgresql://nandhavignesh@localhost:5432/dheera_ai"
dheera-ai --config proxy_server_config.yaml
```

---

## 📚 Configuration

### Basic Model Configuration

Edit `proxy_server_config.yaml`:

```yaml
model_list:
  # OpenAI Models
  - model_name: gpt-4o
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
      timeout: 600
    model_info:
      tags: ["production", "premium"]
      metadata:
        tier: "premium"

  # Anthropic Models
  - model_name: claude-sonnet
    dheera_ai_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      tags: ["production", "analysis"]

  # Local Ollama Models
  - model_name: "ollama/*"
    dheera_ai_params:
      model: "ollama/*"
      api_base: http://localhost:11434
    model_info:
      tags: ["local", "development"]
```

### Available Models

The system includes pre-configured models:
- **OpenAI**: GPT-4o, GPT-3.5-turbo, DALL-E, Embeddings
- **Anthropic**: Claude Sonnet 4.5
- **Google**: Gemini 1.5 Flash
- **Ollama**: phi3:mini, llama2, and any Ollama model (wildcard support)
- **Azure**: Realtime models support
- **And 100+ more providers**

---

## 🔌 API Usage

### Chat Completion
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Use Ollama Local Model
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/phi3:mini",
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
  }'
```

### Embeddings
```bash
curl -X POST http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "Your text here"
  }'
```

### List Models
```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-1234"
```

---

## 🗄️ Vector Stores

DheeraLLM supports vector stores for RAG (Retrieval Augmented Generation):

### Create Vector Store
```bash
curl -X POST http://localhost:4000/v1/vector_stores \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-knowledge-base",
    "metadata": {"project": "dheerallm"}
  }'
```

For comprehensive vector store documentation, see [VECTOR_STORE_GUIDE.md](VECTOR_STORE_GUIDE.md)

---

## 🏷️ Tags & Organization

Use tags to organize and filter models:

```yaml
model_list:
  - model_name: production-gpt4
    dheera_ai_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      tags: ["production", "premium", "chat"]
      metadata:
        department: "engineering"
        cost_center: "ai-ops"
```

Filter models by tags via API:
```bash
curl "http://localhost:4000/v1/models?tags=production,premium" \
  -H "Authorization: Bearer sk-1234"
```

---

## 🔧 Environment Variables

Create a `.env` file or export these variables:

```bash
# Database
export DATABASE_URL="postgresql://user@localhost:5432/dheera_ai"

# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Redis (optional, for caching)
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD=""

# Langfuse (optional, for monitoring)
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
```

---

## 📊 Monitoring & Observability

### Built-in Integrations
- **Langfuse**: Track all LLM calls, costs, and performance
- **Prometheus**: Metrics export for monitoring
- **Custom Callbacks**: Integrate with your own monitoring tools

### Configuration
```yaml
dheera_ai_settings:
  success_callback: ["langfuse", "prometheus"]

  default_team_settings:
    - team_id: team-1
      success_callback: ["langfuse"]
      langfuse_public_key: os.environ/LANGFUSE_PUBLIC_KEY
      langfuse_secret: os.environ/LANGFUSE_SECRET_KEY
```

---

## 🎨 UI Features

Access the admin UI at http://localhost:4000/ui

- **Dashboard**: View usage stats and costs
- **API Keys**: Create and manage API keys
- **Models**: Add, edit, and test models
- **Teams**: Manage teams and permissions
- **Logs**: View detailed request logs
- **Settings**: Configure global settings

---

## 🔐 Security

### Authentication
- **Master Key**: `sk-1234` (change in production!)
- **API Keys**: Create scoped API keys per team/user
- **Password Hashing**: SHA-256 for user passwords

### Best Practices
1. Change the master key in `proxy_server_config.yaml`
2. Use environment variables for API keys
3. Enable HTTPS in production
4. Set up proper database permissions
5. Use team-based access control

---

## 🚀 Production Deployment

### Docker (Coming Soon)
```bash
docker-compose up -d
```

### Kubernetes
Helm charts available in `deploy/charts/dheera_ai-helm/`

### Environment Setup
1. Use a managed PostgreSQL database
2. Enable Redis for caching and routing
3. Set up proper logging and monitoring
4. Configure rate limits and budgets
5. Enable health checks

---

## 📖 Documentation

- **Vector Stores**: [VECTOR_STORE_GUIDE.md](VECTOR_STORE_GUIDE.md)
- **Configuration**: Check `proxy_server_config.yaml` for examples
- **API Reference**: http://localhost:4000/docs (when server is running)
- **Official Docs**: https://docs.dheeraai.com

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Based on the LiteLLM project
- Built with FastAPI, Prisma, and Next.js
- Thanks to all the LLM providers for their amazing APIs

---

## 📞 Support

- **GitHub Issues**: https://github.com/nandha030/dheera_ai/issues
- **Email**: admin@dheeraai.com

---

## 🗺️ Roadmap

- [ ] Docker containerization
- [ ] Additional vector store backends
- [ ] Fine-tuning API support
- [ ] Advanced load balancing strategies
- [ ] Multi-region deployment support
- [ ] Enhanced UI features

---

**Built with ❤️ by Nandhavignesh**

**DheeraLLM v0.3.0** - Making LLM integration simple and powerful
