import os
import json
import requests
import time
import random
import functools
from typing import Dict, Any, Optional, Generator, List, Callable, TypeVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
import urllib3

# Library Imports for LLM Providers
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialization
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()

# Check if API mode is enabled
API_MODE = os.environ.get("LLM_API_MODE", "false").lower() in ("true", "1", "yes", "on")

# Conditional Flask imports and initialization
app = None
request = None
jsonify = None
Response = None
stream_with_context = None

if API_MODE:
    try:
        from flask import Flask, request, jsonify, Response, stream_with_context
        from flask_session import Session

        app = Flask(__name__)
        app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "default_fallback_secret_key")
        app.config["SESSION_PERMANENT"] = False
        app.config["SESSION_TYPE"] = "filesystem"
        Session(app)
    except ImportError as e:
        print(f"⚠ API_MODE is enabled but Flask is not installed: {e}")
        print("  Install Flask with: pip install flask flask-session")
        API_MODE = False


def _register_flask_routes():
    """Register Flask routes only when Flask is available and API mode is enabled."""
    if not (API_MODE and app):
        return

    # CORS Configuration
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

    @app.route('/api/llm/call', methods=['POST'])
    def llm_call():
        """Unified LLM API endpoint"""
        try:
            data = request.json

            req = LLMRequest(
                provider=data.get('provider'),
                model=data.get('model'),
                prompt=data.get('prompt'),
                stream=data.get('stream', False),
                temperature=data.get('temperature'),
                max_tokens=data.get('max_tokens'),
                system_prompt=data.get('system_prompt'),
                messages=data.get('messages'),
                reasoning_effort=data.get('reasoning_effort'),
                enable_thinking=data.get('enable_thinking', True),
                json_mode=data.get('json_mode', False)
            )

            if req.stream:
                return Response(
                    stream_with_context(LLMService.stream(req)),
                    mimetype='text/event-stream'
                )
            else:
                response = LLMService.call(req)
                return jsonify({
                    'content': response.content,
                    'model': response.model,
                    'provider': response.provider,
                    'usage': response.usage,
                    'reasoning_content': response.reasoning_content,
                    'finish_reason': response.finish_reason
                })

        except CircuitBreakerOpenException as cbe:
            return jsonify({'error': str(cbe), 'code': 'CIRCUIT_OPEN'}), 503
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/providers', methods=['GET'])
    def get_providers():
        """Get available providers and their models"""
        providers_info = {}

        for provider, config in PROVIDER_CONFIGS.items():
            if provider == "resilience": continue
            providers_info[provider] = {
                'models': list(config.get('models', {}).keys()),
                'default_model': config.get('default_model'),
                'available': True,
                'circuit_status': resilience_manager.get_breaker(provider).state
            }

        if os.environ.get("OLLAMA_CHAT_ENDPOINT"):
            ollama_models = get_ollama_models_from_api()
            providers_info['ollama'] = {
                'models': [m['id'] for m in ollama_models],
                'default_model': ollama_models[0]['id'] if ollama_models else None,
                'available': len(ollama_models) > 0,
                'circuit_status': resilience_manager.get_breaker('ollama').state
            }

        return jsonify(providers_info)

    @app.route('/api/providers/<provider>/models', methods=['GET'])
    def get_provider_models(provider):
        if provider == "ollama":
            ollama_models = get_ollama_models_from_api()
            if not ollama_models:
                return jsonify({'error': 'Ollama not configured or no models available'}), 500

            models = []
            for model in ollama_models:
                models.append({
                    'name': model['id'],
                    'max_tokens': 8192,
                    'supports_streaming': True,
                    'supports_reasoning': False,
                    'supports_extended_thinking': False,
                    'default_temperature': 0.3,
                    'knowledge_cutoff': None
                })

            return jsonify({
                'provider': provider,
                'models': models,
                'default_model': ollama_models[0]['id'] if ollama_models else None
            })

        config = PROVIDER_CONFIGS.get(provider)
        if not config:
            return jsonify({'error': f'Unknown provider: {provider}'}), 404

        models = []
        for model_name, model_data in config.get('models', {}).items():
            model_config = ModelConfig.from_dict(model_data)
            models.append({
                'name': model_name,
                'max_tokens': model_config.max_tokens,
                'supports_streaming': model_config.supports_streaming,
                'supports_reasoning': model_config.supports_reasoning,
                'supports_extended_thinking': model_config.supports_extended_thinking,
                'default_temperature': model_config.temperature_default,
                'knowledge_cutoff': model_config.knowledge_cutoff
            })

        return jsonify({
            'provider': provider,
            'models': models,
            'default_model': config.get('default_model')
        })

    @app.route('/api/config/reload', methods=['POST'])
    def reload_config():
        global PROVIDER_CONFIGS
        try:
            config_file = request.json.get('config_file') if request.json else None
            PROVIDER_CONFIGS = load_provider_config(config_file or CONFIG_FILE)
            resilience_manager.config = PROVIDER_CONFIGS.get("resilience", {})
            return jsonify({
                'status': 'success',
                'message': 'Configuration reloaded',
                'providers': list(PROVIDER_CONFIGS.keys())
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'providers': {
                'openai': clients.openai is not None,
                'azure': clients.azure is not None,
                'anthropic': clients.anthropic is not None,
                'gemini': clients.gemini is not None,
                'deepseek': os.environ.get("DEEPSEEK_API_KEY") is not None,
                'xai': os.environ.get("XAI_API_KEY") is not None,
                'perplexity': os.environ.get("PERPLEXITY_API_KEY") is not None,
                'ollama': os.environ.get("OLLAMA_CHAT_ENDPOINT") is not None
            },
            'circuit_breakers': {
                 name: breaker.state for name, breaker in resilience_manager.breakers.items()
            }
        })


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_provider_config(config_path: Optional[str] = None) -> Dict:
    """Load provider configuration."""
    if config_path and not Path(config_path).exists():
        sibling_path = Path(__file__).parent / config_path
        if sibling_path.exists():
            config_path = str(sibling_path)

    if config_path is None:
        possible_paths = [
            Path("llm_config.json"),
            Path("config/llm_config.json"),
            Path(__file__).parent / "llm_config.json",
            Path(__file__).parent / "config" / "llm_config.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "resilience" not in config:
                    config["resilience"] = get_default_config()["resilience"]
                print(f"✓ Loaded configuration from {config_path}")
                return config
        except Exception as e:
            print(f"⚠ Error loading config from {config_path}: {e}")
            print("  Using default configuration")
    else:
        print("⚠ No config file found. Using default configuration")
    
    return get_default_config()


def get_default_config() -> Dict:
    """Default configuration as fallback"""
    return {
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
                    "supports_streaming": True,
                    "supports_reasoning": False,
                    "temperature_default": 0.3
                }
            }
        },
        "anthropic": {
            "api_version": "2023-06-01",
            "default_model": "claude-sonnet-4.5-20250929",
            "models": {
                "claude-sonnet-4.5-20250929": {
                    "max_tokens": 8192,
                    "supports_streaming": True,
                    "supports_reasoning": False,
                    "temperature_default": 0.3
                }
            }
        }
    }


CONFIG_FILE = os.environ.get("LLM_CONFIG_FILE", "llm_config.json")
PROVIDER_CONFIGS = load_provider_config(CONFIG_FILE)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class Provider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    AZURE = "azure_oai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    max_tokens: int
    supports_streaming: bool
    supports_reasoning: bool = False
    temperature_default: float = 0.3
    uses_completion_tokens: bool = False
    supports_extended_thinking: bool = False
    knowledge_cutoff: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        return cls(
            max_tokens=data.get('max_tokens', 8192),
            supports_streaming=data.get('supports_streaming', True),
            supports_reasoning=data.get('supports_reasoning', False),
            temperature_default=data.get('temperature_default', 0.3),
            uses_completion_tokens=data.get('uses_completion_tokens', False),
            supports_extended_thinking=data.get('supports_extended_thinking', False),
            knowledge_cutoff=data.get('knowledge_cutoff')
        )


def get_ollama_models_from_api() -> List[Dict[str, Any]]:
    """Fetch models from Ollama API endpoint"""
    endpoint = os.environ.get("OLLAMA_MODELS_ENDPOINT")
    api_key = os.environ.get("OLLAMA_API_KEY")
    
    if not endpoint or not api_key:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(endpoint, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            return [{"id": model.get("id"), "name": model.get("id")} for model in data.get("data", []) if model.get("id")]
        elif "models" in data:
            return [{"id": model.get("name"), "name": model.get("name")} for model in data.get("models", []) if model.get("name")]
        else:
            return []
    except Exception as e:
        print(f"⚠ Error fetching Ollama models: {e}")
        return []


def get_model_config(provider: str, model: str) -> ModelConfig:
    """Get model configuration from loaded config or Ollama API"""
    if provider == "ollama":
        return ModelConfig(8192, True, False, 0.3)
    
    provider_config = PROVIDER_CONFIGS.get(provider, {})
    models = provider_config.get('models', {})
    model_data = models.get(model, {})
    
    if not model_data:
        return ModelConfig(8192, True, False, 0.3)
    
    return ModelConfig.from_dict(model_data)


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

class LLMClients:
    """Centralized LLM client management"""
    
    def __init__(self):
        self.openai = None
        self.azure = None
        self.anthropic = None
        self.gemini = None
        
        if os.environ.get("OPENAI_API_KEY"):
            self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print("✓ OpenAI client initialized")
        
        if os.environ.get("AZURE_OAI_ENDPOINT") and os.environ.get("AZURE_OAI_KEY"):
            self.azure = AzureOpenAI(
                api_key=os.environ.get("AZURE_OAI_KEY"),
                azure_endpoint=os.environ.get("AZURE_OAI_ENDPOINT"),
                api_version="2024-10-21"
            )
            self.azure_deployment = os.environ.get("AZURE_OAI_DEPLOYMENT_NAME")
            print("✓ Azure OpenAI client initialized")
        
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            print("✓ Anthropic client initialized")
        
        try:
            if os.environ.get("GEMINI_API_KEY"):
                self.gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                print("✓ Gemini client initialized")
        except Exception as e:
            print(f"⚠ Could not configure Gemini: {e}")


clients = LLMClients()


# ============================================================================
# UNIFIED LLM INTERFACE
# ============================================================================

@dataclass
class LLMRequest:
    """Unified request structure for all providers"""
    provider: str
    model: str
    prompt: str
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    reasoning_effort: Optional[str] = None
    enable_thinking: bool = True
    json_mode: bool = False  


@dataclass
class LLMResponse:
    """Unified response structure"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None


# ============================================================================
# RESILIENCE: CIRCUIT BREAKER & RETRY LOGIC
# ============================================================================

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int, recovery_timeout: int):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                print(f"⚡ Circuit Breaker '{self.name}': Entering HALF_OPEN state")
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            print(f"⚡ Circuit Breaker '{self.name}': Closing circuit (Recovered)")
            self.state = "CLOSED"
        self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        print(f"⚠ Circuit Breaker '{self.name}': Failure recorded ({self.failure_count}/{self.failure_threshold})")
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            print(f"🔥 Circuit Breaker '{self.name}': OPENED. Blocking requests for {self.recovery_timeout}s")


class ResilienceManager:
    def __init__(self):
        self.config = PROVIDER_CONFIGS.get("resilience", {})
        self.breakers: Dict[str, CircuitBreaker] = {}
        
    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        if provider_name not in self.breakers:
            self.breakers[provider_name] = CircuitBreaker(
                name=provider_name,
                failure_threshold=self.config.get("circuit_breaker_failure_threshold", 5),
                recovery_timeout=self.config.get("circuit_breaker_recovery_timeout", 60)
            )
        return self.breakers[provider_name]

    def execute_with_resilience(self, func: Callable, request: LLMRequest, *args, **kwargs):
        provider_name = request.provider
        breaker = self.get_breaker(provider_name)
        
        if not breaker.can_execute():
            remaining = int(breaker.recovery_timeout - (time.time() - breaker.last_failure_time))
            raise CircuitBreakerOpenException(f"Provider '{provider_name}' is currently unavailable. Try again in {remaining}s.")

        max_retries = self.config.get("max_retries", 3)
        backoff_factor = self.config.get("backoff_factor", 1.5)
        jitter = self.config.get("retry_jitter", 0.5)

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            
            except Exception as e:
                is_retryable = self._is_retryable_error(e)
                last_exception = e
                
                if not is_retryable or attempt == max_retries:
                    breaker.record_failure()
                    raise e
                
                sleep_time = (backoff_factor ** attempt) + random.uniform(0, jitter)
                print(f"↺ Retry {attempt + 1}/{max_retries} for '{provider_name}' in {sleep_time:.2f}s. Reason: {str(e)}")
                time.sleep(sleep_time)
        
        if last_exception:
            raise last_exception

    def _is_retryable_error(self, e: Exception) -> bool:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg: return False
        if "400" in error_msg and "bad request" in error_msg: return False
        if "invalid_api_key" in error_msg: return False
        if "model not found" in error_msg: return False
        
        if "429" in error_msg or "rate limit" in error_msg: return True
        if "500" in error_msg or "502" in error_msg or "503" in error_msg: return True
        if "timeout" in error_msg: return True
        if "connection" in error_msg: return True
        if "overloaded" in error_msg: return True
        return False

resilience_manager = ResilienceManager()


# ============================================================================
# PROVIDER IMPLEMENTATIONS
# ============================================================================

class OpenAIProvider:
    """OpenAI and compatible providers"""
    
    @staticmethod
    def call(req: LLMRequest, client, api_base: str = None, extra_body: dict = None) -> LLMResponse:
        """Unified OpenAI-compatible API call

        Args:
            req: LLMRequest object
            client: OpenAI client instance
            api_base: Optional API base URL (for backward compatibility)
            extra_body: Optional extra_body dict for provider-specific parameters (e.g., DeepSeek thinking)

        Returns:
            LLMResponse object
        """
        config = get_model_config(req.provider, req.model)

        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt}]

        # Handle System Prompt Merging
        if req.system_prompt and not any(m.get("role") == "system" for m in messages):
            # For O1/Reasoning models, some don't support system, but we handle it generally here
            if req.model.startswith("o1") or req.model.startswith("gpt-5"):
                 # O1 beta sometimes restricted system prompt, but generally supported now as "developer" or "system"
                 # depending on version. We'll stick to 'system' for broad compatibility or merge into user.
                 messages.insert(0, {"role": "system", "content": req.system_prompt})
            else:
                 messages.insert(0, {"role": "system", "content": req.system_prompt})

        params = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream
        }

        # JSON Mode Support
        if req.json_mode:
            params["response_format"] = {"type": "json_object"}
            # Ensure "JSON" is mentioned in the prompt, otherwise OpenAI throws 400
            json_instruction = "You are a helpful assistant designed to output JSON."
            if messages[0]["role"] == "system":
                messages[0]["content"] += f" {json_instruction}"
            else:
                messages.insert(0, {"role": "system", "content": json_instruction})

        # Reasoning Model Handling (o1, etc)
        if config.supports_reasoning:
            params["max_completion_tokens"] = req.max_tokens or config.max_tokens
            if req.reasoning_effort:
                params["reasoning_effort"] = req.reasoning_effort
        else:
            params["max_tokens"] = req.max_tokens or config.max_tokens
            params["temperature"] = req.temperature if req.temperature is not None else config.temperature_default

        # Add extra_body if provided (for provider-specific params like DeepSeek thinking)
        if extra_body:
            params.update(extra_body)

        try:
            response = client.chat.completions.create(**params)
            
            if req.stream:
                return response
            
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
            return LLMResponse(
                content=content,
                model=req.model,
                provider=req.provider,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') else None,
                reasoning_content=reasoning,
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            raise e
    
    @staticmethod
    def stream(req: LLMRequest, client) -> Generator[str, None, None]:
        stream = OpenAIProvider.call(req, client)
        reasoning_buffer = []

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'chunk': chunk.choices[0].delta.content})}\n\n"

            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_buffer.append(chunk.choices[0].delta.reasoning_content)
                if len(reasoning_buffer) >= 5:
                    yield f"data: {json.dumps({'reasoning': ''.join(reasoning_buffer)})}\n\n"
                    reasoning_buffer = []

        if reasoning_buffer:
            yield f"data: {json.dumps({'reasoning': ''.join(reasoning_buffer)})}\n\n"

        yield "data: [DONE]\n\n"


class AnthropicProvider:
    """Anthropic Claude API"""
    
    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        if not clients.anthropic:
            raise Exception("Anthropic client not configured")
        
        config = get_model_config(req.provider, req.model)
        
        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt}]
        
        # JSON Mode Handling for Anthropic
        # Technique: Append instruction to system prompt AND prefill assistant message
        if req.json_mode:
            req.system_prompt = (req.system_prompt or "") + "\n\nIMPORTANT: Return ONLY valid JSON. Do not include any explanation or markdown formatting."
            
            # Prefill assistant message to force JSON start
            # Check if last message is assistant; if so, append. If not, add new assistant message.
            if messages and messages[-1]["role"] == "assistant":
                # Already has assistant response? This is rare in a fresh request, usually it ends with user.
                pass 
            else:
                 # Note: Anthropic allows pre-filling the Assistant response to guide output
                 # However, the SDK expects 'messages' to be alternating User/Assistant. 
                 # We cannot easily "prefill" in the API request object unless we want the model to continue from that prefill.
                 # If we add {"role": "assistant", "content": "{"}, the model continues.
                 # We will use this robust technique.
                 messages.append({"role": "assistant", "content": "{"})

        params = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens or config.max_tokens,
            "temperature": req.temperature if req.temperature is not None else config.temperature_default,
            "stream": req.stream
        }
        
        if req.system_prompt:
            params["system"] = req.system_prompt
        
        if config.supports_extended_thinking and req.enable_thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }
        
        response = clients.anthropic.messages.create(**params)
        
        if req.stream:
            return response
        
        content_text = ""
        thinking_text = ""
        
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                thinking_text += block.thinking
        
        # If we prefilled "{", we need to prepend it back to the result to make valid JSON
        if req.json_mode and messages[-1]["content"] == "{":
            content_text = "{" + content_text

        return LLMResponse(
            content=content_text,
            model=req.model,
            provider=req.provider,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            reasoning_content=thinking_text if thinking_text else None,
            finish_reason=response.stop_reason
        )
    
    @staticmethod
    def stream(req: LLMRequest) -> Generator[str, None, None]:
        # Handle the prefill logic for streaming too
        prefilled_brace = False
        if req.json_mode:
             # Similar logic to call()
             if req.messages:
                 if req.messages[-1]["role"] != "assistant":
                     req.messages.append({"role": "assistant", "content": "{"})
                     prefilled_brace = True
             else:
                 # Default prompt flow
                 req.messages = [{"role": "user", "content": req.prompt}, {"role": "assistant", "content": "{"}]
                 prefilled_brace = True
                 
             req.system_prompt = (req.system_prompt or "") + "\n\nIMPORTANT: Return ONLY valid JSON."

        stream = AnthropicProvider.call(req)
        reasoning_buffer = []

        # If we prefilled, yield the brace first
        if prefilled_brace:
            yield f"data: {json.dumps({'chunk': '{'})}\n\n"

        with stream as event_stream:
            for event in event_stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield f"data: {json.dumps({'chunk': event.delta.text})}\n\n"

                    if hasattr(event.delta, 'thinking'):
                        reasoning_buffer.append(event.delta.thinking)
                        if len(reasoning_buffer) >= 5:
                            yield f"data: {json.dumps({'reasoning': ''.join(reasoning_buffer)})}\n\n"
                            reasoning_buffer = []

        if reasoning_buffer:
            yield f"data: {json.dumps({'reasoning': ''.join(reasoning_buffer)})}\n\n"

        yield "data: [DONE]\n\n"


class GeminiProvider:
    """Google Gemini API with new google-genai SDK"""
    
    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        if not clients.gemini:
            raise Exception("Gemini not configured")
        
        config = get_model_config(req.provider, req.model)
        
        generation_config = genai_types.GenerateContentConfig(
            temperature=req.temperature if req.temperature is not None else config.temperature_default,
            system_instruction=req.system_prompt if req.system_prompt else None
        )
        
        # JSON Mode
        if req.json_mode:
            generation_config.response_mime_type = "application/json"
        
        if config.supports_reasoning and req.enable_thinking:
            generation_config.thinking_config = {"mode": "deep"}
        
        if req.messages:
            contents = []
            for msg in req.messages:
                role = "user" if msg["role"] in ["user", "system"] else "model"
                contents.append(genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=msg["content"])]
                ))
        else:
            contents = req.prompt
        
        if req.stream:
            response = clients.gemini.models.generate_content_stream(
                model=req.model,
                contents=contents,
                config=generation_config
            )
            return response
        else:
            response = clients.gemini.models.generate_content(
                model=req.model,
                contents=contents,
                config=generation_config
            )
            
            content_text = ""
            if hasattr(response, 'text'):
                content_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        content_text += part.text
            
            return LLMResponse(
                content=content_text,
                model=req.model,
                provider=req.provider,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                } if hasattr(response, 'usage_metadata') and response.usage_metadata else None,
                finish_reason=response.candidates[0].finish_reason if hasattr(response, 'candidates') and response.candidates else None
            )
    
    @staticmethod
    def stream(req: LLMRequest) -> Generator[str, None, None]:
        stream = GeminiProvider.call(req)
        
        for chunk in stream:
            text_content = ""
            if hasattr(chunk, 'text') and chunk.text:
                text_content = chunk.text
            elif hasattr(chunk, 'candidates') and chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text
            
            if text_content:
                yield f"data: {json.dumps({'chunk': text_content})}\n\n"
        
        yield "data: [DONE]\n\n"


class DeepSeekProvider:
    """DeepSeek API with R1 reasoning support

    When enable_thinking=True:
    - For deepseek-chat: API automatically routes to deepseek-reasoner internally
    - For deepseek-reasoner: Thinking is automatic by default
    Response includes reasoning_content with thinking tokens.

    Note: DeepSeek does not support effort levels (unlike OpenAI's reasoning_effort).
    Thinking is either enabled or disabled via enable_thinking boolean.
    """

    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise Exception("DeepSeek API key not configured")

        config = get_model_config(req.provider, req.model)

        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt}]
        if req.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": req.system_prompt})

        # DeepSeek specific reasoning setup
        # For deepseek-chat: use extra_body to enable thinking tokens (API routes to deepseek-reasoner)
        # For deepseek-reasoner: thinking is automatic, no extra_body needed
        extra_body = None
        if config.supports_reasoning and req.enable_thinking:
            # Only add extra_body for deepseek-chat (R1 models have automatic reasoning)
            if not req.model.startswith("deepseek-reasoner"):
                extra_body = {"thinking": {"type": "enabled"}}

        req.messages = messages

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        # Reuse OpenAI Logic which handles json_mode = response_format={"type": "json_object"}
        return OpenAIProvider.call(req, client, extra_body=extra_body)


class XAIProvider:
    """xAI Grok API with reasoning support

    xAI's Grok API is OpenAI-compatible and supports models like:
    - grok-beta, grok-3: Standard models
    - grok-4: Reasoning model (always in reasoning mode, no non-reasoning option)

    Note: Grok 4 does not support reasoning_effort parameter. Reasoning is automatic.
    Supported parameters: presence_penalty, frequency_penalty, stop (not for grok-4).
    """

    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise Exception("xAI API key not configured. Set XAI_API_KEY environment variable.")

        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt}]
        if req.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": req.system_prompt})

        req.messages = messages

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )

        # Reuse OpenAI Logic which handles json_mode = response_format={"type": "json_object"}
        # Note: Grok 4 reasoning_content will be captured if present in response
        return OpenAIProvider.call(req, client)


class OllamaProvider:
    """Custom Ollama endpoint (OpenAI-compatible)

    Supports both authenticated and unauthenticated Ollama instances:
    - Local Ollama: typically doesn't require authentication
    - Remote Ollama: may require OLLAMA_API_KEY for authentication

    Only OLLAMA_CHAT_ENDPOINT is required. OLLAMA_API_KEY is optional.
    """

    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        endpoint = os.environ.get("OLLAMA_CHAT_ENDPOINT")
        api_key = os.environ.get("OLLAMA_API_KEY")

        if not endpoint:
            raise Exception("Ollama endpoint not configured. Set OLLAMA_CHAT_ENDPOINT environment variable.")

        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt}]
        if req.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": req.system_prompt})

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream,
            "temperature": req.temperature or 0.3,
        }
        
        if req.json_mode:
            payload["format"] = "json"

        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            verify=False,
            timeout=300,
            stream=req.stream
        )
        response.raise_for_status()
        
        if req.stream:
            return response
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return LLMResponse(
            content=content,
            model=req.model,
            provider=req.provider,
            usage=data.get("usage"),
            finish_reason=data["choices"][0].get("finish_reason")
        )
    
    @staticmethod
    def stream(req: LLMRequest) -> Generator[str, None, None]:
        response = OllamaProvider.call(req)
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                    except json.JSONDecodeError:
                        continue
        
        yield "data: [DONE]\n\n"


# ============================================================================
# EXCEPTION CLASSES (Add this above LLMService)
# ============================================================================

class EmptyLLMResponseError(Exception):
    """Raised when the LLM provider returns an empty response (often due to safety filters)"""
    pass

# ============================================================================
# UNIFIED LLM SERVICE
# ============================================================================

class LLMService:
    """Unified interface for all LLM providers"""
    
    @staticmethod
    def _get_provider_implementation(req: LLMRequest):
        provider = req.provider.lower()
        
        if provider == "openai":
            if not clients.openai: raise Exception("OpenAI client not configured")
            return lambda: OpenAIProvider.call(req, clients.openai)
        
        elif provider == "azure_oai":
            if not clients.azure: raise Exception("Azure OpenAI client not configured")
            req.model = clients.azure_deployment
            return lambda: OpenAIProvider.call(req, clients.azure)
        
        elif provider == "anthropic":
            return lambda: AnthropicProvider.call(req)
        
        elif provider == "gemini":
            return lambda: GeminiProvider.call(req)
        
        elif provider == "deepseek":
            return lambda: DeepSeekProvider.call(req)

        elif provider == "xai":
            return lambda: XAIProvider.call(req)

        elif provider == "perplexity":
            api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not api_key: raise Exception("Perplexity API key not configured")
            client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            return lambda: OpenAIProvider.call(req, client)

        elif provider == "ollama":
            return lambda: OllamaProvider.call(req)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def call(req: LLMRequest) -> LLMResponse:
        if req.stream:
            raise ValueError("Use stream() method for streaming requests")
        target_func = LLMService._get_provider_implementation(req)
        
        # 1. Execute via Resilience Manager
        response = resilience_manager.execute_with_resilience(target_func, req)
        
        # 2. VALIDATION: Check for empty response (Safety Filters/Glitches)
        if not response.content or not str(response.content).strip():
             raise EmptyLLMResponseError(
                 f"Provider '{req.provider}' returned an empty response. "
                 "This often indicates a Safety Filter trigger or a model failure."
             )

        # 3. LOGGING: Safe extraction of metadata
        # usage: getattr(obj, name, default) handles if the attribute is missing entirely
        # 'or "unknown"' handles if the attribute exists but is None
        finish_reason = getattr(response, 'finish_reason', None) or "unknown"
        usage = getattr(response, 'usage', None) or {}
        total_tokens = usage.get('total_tokens', 'N/A')

        # Log to console (or replace with app.logger.info)
        print(f"📝 [LLM Log] {req.provider}/{req.model} | Stop: {finish_reason} | Tokens: {total_tokens}")

        return response
    
    @staticmethod
    def stream(req: LLMRequest) -> Generator[str, None, None]:
        req.stream = True
        provider = req.provider.lower()

        def establish_stream():
            if provider == "openai":
                if not clients.openai: raise Exception("OpenAI client not configured")
                return OpenAIProvider.stream(req, clients.openai)
            elif provider == "azure_oai":
                if not clients.azure: raise Exception("Azure OpenAI client not configured")
                req.model = clients.azure_deployment
                return OpenAIProvider.stream(req, clients.azure)
            elif provider == "anthropic":
                return AnthropicProvider.stream(req)
            elif provider == "gemini":
                return GeminiProvider.stream(req)
            elif provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key: raise Exception("DeepSeek API key not configured")
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                return OpenAIProvider.stream(req, client)
            elif provider == "xai":
                api_key = os.environ.get("XAI_API_KEY")
                if not api_key: raise Exception("xAI API key not configured")
                client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                return OpenAIProvider.stream(req, client)
            elif provider == "perplexity":
                api_key = os.environ.get("PERPLEXITY_API_KEY")
                if not api_key: raise Exception("Perplexity API key not configured")
                client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
                return OpenAIProvider.stream(req, client)
            elif provider == "ollama":
                return OllamaProvider.stream(req)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        stream_generator = resilience_manager.execute_with_resilience(establish_stream, req)
        yield from stream_generator


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Register Flask routes when API mode is enabled
_register_flask_routes()


def run_api_server(host: str = '0.0.0.0', port: int = 8888, debug: bool = True):
    """
    Run the Flask API server. Only works when API mode is enabled.

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode

    Raises:
        RuntimeError: If API mode is not enabled or Flask is not available
    """
    if not (API_MODE and app):
        raise RuntimeError(
            "API server is not available. Set LLM_API_MODE=true environment variable "
            "and install Flask: pip install flask flask-session"
        )
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    if API_MODE and app:
        app.run(debug=True, port=8888)
    else:
        print("⚠ API mode is not enabled. Set LLM_API_MODE=true to run the API server.")
        print("  You can still use this module as a library for LLM services.")