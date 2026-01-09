# LLM Services

<div align="center">

**Unified API for Multiple LLM Providers**

Build once, run anywhere: One API for all your LLMs, cloud or local

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.9.0-orange)]https://github.com/ngstcf/llmbase)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://c3.unu.edu/projects/ai/llmbase/)

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 🌟 Key Features

- **🔄 Multi-Provider Support**: OpenAI, Azure OpenAI, Anthropic, Google Gemini, DeepSeek, xAI/Grok, Perplexity, Ollama
- **📦 Structured Output**: Built-in `json_mode` ensures valid JSON responses across all providers
- **🔌 Dual Mode**: Use as a Python library (no Flask) or HTTP API server (optional Flask)
- **🛡️ Resilience**: Automatic retries with exponential backoff and circuit breakers
- **🧠 Advanced Features**: Support for reasoning models, streaming, extended thinking
- **🔧 Configuration-Driven**: Hot-reload model configs without code changes
- **🌐 CORS-Ready**: Built-in CORS support for web applications
- **🔍 Debugging & Transparency**: Built-in logging, request tracking, performance metrics, and configuration status endpoints

---

## 📥 Installation

### Install from GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/llmbase.git
cd llmbase

# Install dependencies
pip install -r requirements.txt
```

### Or install with pip directly from GitHub

```bash
pip install git+https://github.com/yourusername/llmbase.git
```

### Dependencies

**Core (Library Mode - no Flask required):**
- python-dotenv
- openai
- anthropic
- google-genai
- requests
- urllib3

**Optional (API Server Mode):**
- flask
- flask-session

Enable API server mode by setting `LLM_API_MODE=true` in your `.env` file.

---

## 🚀 Quick Start

### As a Python Library

```python
from llmservices import LLMService, LLMRequest

# Simple call
req = LLMRequest(
    provider="openai",
    model="gpt-4o",
    prompt="Write a haiku about AI"
)
response = LLMService.call(req)
print(response.content)
```

### As an API Server

```bash
# Enable API mode
export LLM_API_MODE=true

# Run the server
python llmservices.py
```

Then make HTTP requests:

```bash
curl -X POST http://localhost:8888/api/llm/call \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "prompt": "Hello, world!"
  }'
```

---

## 📚 Documentation

**📖 Full Documentation**: [https://c3.unu.edu/projects/ai/llmbase/](https://c3.unu.edu/projects/ai/llmbase/)

**📝 Blog Post**: [One API, Many AI Models](https://c3.unu.edu/blog/llmbase-one-api-many-ai-models)

The complete HTML documentation is hosted online with:
- Interactive navigation
- Code examples
- API reference
- Best practices guide

### Environment Configuration

Create a `.env` file:

```bash
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...

# Optional Providers
PERPLEXITY_API_KEY=pplx-...
AZURE_OAI_ENDPOINT=https://...
AZURE_OAI_KEY=...
AZURE_OAI_DEPLOYMENT_NAME=gpt-4

# Ollama (Local)
OLLAMA_CHAT_ENDPOINT=http://localhost:11434/api/chat
OLLAMA_MODELS_ENDPOINT=http://localhost:11434/api/models

# Service Config
LLM_CONFIG_FILE=llm_config.json
LLM_API_MODE=false  # Set to true for API server mode
FLASK_SECRET_KEY=your-secret-key

# Debugging & Logging (Optional)
LLM_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LLM_DEBUG=false     # Enable verbose logging for debugging
```

### Model Configuration

Create `llm_config.json`:

```json
{
  "resilience": {
    "max_retries": 3,
    "backoff_factor": 1.5,
    "retry_jitter": 0.5,
    "circuit_breaker_failure_threshold": 5,
    "circuit_breaker_recovery_timeout": 60
  },
  "openai": {
    "api_base": "https://api.openai.com/v1",
    "default_model": "gpt-4o",
    "models": {
      "gpt-4o": {
        "max_tokens": 16384,
        "supports_streaming": true,
        "temperature_default": 0.3,
        "uses_completion_tokens": false
      },
      "o1": {
        "max_tokens": 100000,
        "supports_streaming": false,
        "supports_reasoning": true,
        "uses_completion_tokens": true
      }
    }
  },
  "anthropic": {
    "default_model": "claude-sonnet-4-5-20250929",
    "models": {
      "claude-sonnet-4-5-20250929": {
        "max_tokens": 8192,
        "supports_streaming": true,
        "supports_extended_thinking": true,
        "uses_completion_tokens": false
      }
    }
  }
}
```

> **Note**: The `uses_completion_tokens` field indicates whether the model uses `max_completion_tokens` instead of `max_tokens` in the API request (e.g., OpenAI o1 series). Set to `true` for reasoning models that require this parameter.

### Supported Providers

| Provider | Description |
|----------|-------------|
| OpenAI | GPT models |
| Azure OpenAI | GPT models via Azure |
| Anthropic | Claude models |
| Google Gemini | Gemini models |
| DeepSeek | Chat and reasoning models |
| xAI / Grok | Grok models |
| Perplexity | Sonar models |
| Ollama | Local models |

---

## 💡 Examples

### JSON Mode

```python
req = LLMRequest(
    provider="openai",
    model="gpt-4o",
    prompt="Extract names from: Meeting with Sarah on 2025-05-12",
    json_mode=True
)
response = LLMService.call(req)
import json
data = json.loads(response.content)
print(data)  # {"names": ["Sarah"]}
```

### Streaming

```python
req = LLMRequest(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    prompt="Write a short story",
    stream=True
)

for chunk in LLMService.stream(req):
    print(chunk, end='', flush=True)
```

### Error Handling

```python
from llmservices import CircuitBreakerOpenException, LLMError

try:
    response = LLMService.call(req)
except LLMError as e:
    # Enhanced error with context
    print(f"Error: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Status: {e.status_code}")
    print(f"Request ID: {e.request_id}")
except CircuitBreakerOpenException as e:
    print(f"Service unavailable: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### Debugging

```python
from llmservices import LLMService, LLMRequest, LLMConfig

# Check configuration status
status = LLMConfig.get_status()
print(f"Version: {status['version']}")
print(f"Providers: {status['providers_configured']}")

# Request with tracking
req = LLMRequest(
    provider="openai",
    model="gpt-4o",
    prompt="Hello"
)
print(f"Request ID: {req.request_id}")

response = LLMService.call(req)

# Access debugging information
print(f"Request ID: {response.request_id}")
print(f"Usage: {response.usage}")
print(f"Finish Reason: {response.finish_reason}")
if response.timing:
    print(f"Duration: {response.timing.total_duration_ms}ms")
```

---

## 📖 API Reference

### LLMRequest

```python
@dataclass
class LLMRequest:
    provider: str                      # Required: Provider name
    model: str                         # Required: Model name
    prompt: str                        # Required: User prompt
    stream: bool = False               # Enable streaming
    temperature: Optional[float] = None # 0.0-1.0
    max_tokens: Optional[int] = None   # Max response tokens
    system_prompt: Optional[str] = None  # System message
    messages: Optional[List[Dict]] = None  # Chat messages
    reasoning_effort: Optional[str] = None # "low", "medium", "high"
    enable_thinking: bool = True       # Enable extended thinking
    json_mode: bool = False            # Force JSON output
```

### LLMResponse

```python
@dataclass
class LLMResponse:
    content: str                       # Response text
    model: str                         # Model used
    provider: str                      # Provider used
    usage: Optional[Dict[str, int]]    # Token usage
    reasoning_content: Optional[str]   # Thinking content
    finish_reason: Optional[str]       # Stop reason
    # Debugging fields
    request_id: str                    # Unique request identifier
    response_headers: Optional[Dict[str, str]]  # HTTP response headers
    rate_limit_remaining: Optional[int]         # Rate limit info
    timing: Optional[LLMTiming]        # Performance metrics
    metadata: Optional[LLMMetadata]    # Request metadata
```

### LLMService Methods

```python
class LLMService:
    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        """Make a non-streaming LLM call"""

    @staticmethod
    def stream(req: LLMRequest) -> Generator[str, None, None]:
        """Make a streaming LLM call"""
```

---

## 🔧 Resilience Features

### Automatic Retry

Configured in `llm_config.json`:

```json
{
  "resilience": {
    "max_retries": 3,
    "backoff_factor": 1.5,
    "retry_jitter": 0.5
  }
}
```

### Circuit Breaker

Automatically blocks failing providers:

- **CLOSED**: Normal operation
- **OPEN**: Blocking requests (after threshold failures)
- **HALF_OPEN**: Testing recovery (after timeout)

---

## 🌐 API Endpoints (Server Mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/llm/call` | POST | Make LLM call |
| `/api/providers` | GET | List providers |
| `/api/providers/<provider>/models` | GET | List models |
| `/api/config/reload` | POST | Reload config |
| `/api/config/status` | GET | Get configuration status |
| `/health` | GET | Health check with detailed status |

---

## 📦 Project Structure

```
llmbase/
├── llmservices.py      # Main library
├── llm_config.json     # Model configuration
├── .env                # Environment secrets
├── .env.example        # Environment template
├── requirements.txt    # Dependencies
├── README.md           # This file
├── LICENSE             # MIT License
├── .gitignore          # Git ignore rules
└── examples/           # Usage examples
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ng Chong**

- GitHub: [@ngstcf](https://github.com/ngstcf)

---

## 📝 Changelog

### v1.9.0 (Debugging & Transparency)
- Added structured logging system with configurable log levels
- Added `LLMMetadata` class for request/response tracking
- Added `LLMTiming` class for performance metrics
- Added `LLMError` class with enhanced error context
- Added `LLMConfig.get_status()` for configuration transparency
- Added debug mode (`LLM_DEBUG`) for verbose logging
- Enhanced health check endpoint with detailed status
- Added `/api/config/status` endpoint for debugging
- Added `request_id` tracking for all requests
- Updated `LLMResponse` with debugging fields

### v1.8.0 (xAI Grok Support)
- Added xAI/Grok provider support
- Added Grok 4 reasoning model support
- Fixed Ollama API key to be optional

### v1.7.0 (DeepSeek Reasoning)
- Added DeepSeek provider support
- Added DeepSeek reasoning model (R1) support
- Added thinking tokens support via extra_body

### v1.6.0 (Conditional Flask - Library Mode)
- Added conditional Flask imports
- Library mode now works without Flask
- Added `LLM_API_MODE` environment variable
- Added `run_api_server()` function

### v1.5.0 (JSON Mode Update)
- Added `json_mode` support for all providers
- Improved JSON output handling

---

<div align="center">

**Built for the AI community**

[⬆ Back to Top](#llm-services)

</div>
