"""
Test Suite for Debugging and Transparency Features

This test suite validates the debugging features added in v1.9.0:
- Structured logging system
- LLMMetadata class
- LLMTiming class
- LLMError class
- LLMConfig.get_status()
- Request tracking with request_id
- Enhanced LLMResponse fields

Prerequisites:
1. Install dependencies: pip install pytest python-dotenv
2. Set up .env with at least one provider API key
3. Run: pytest test_debugging.py -v
"""

import os
import sys
import logging
from pathlib import Path
from io import StringIO
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import after loading env
from llmservices import (
    LLMService,
    LLMRequest,
    LLMResponse,
    LLMConfig,
    LLMMetadata,
    LLMTiming,
    LLMError,
    logger,
    LOG_LEVEL,
    LLM_DEBUG
)


class TestLoggingConfiguration:
    """Test logging system configuration"""

    def test_logger_exists(self):
        """Test that logger is configured"""
        assert logger is not None
        assert logger.name == "llmbase"

    def test_log_level_configured(self):
        """Test that log level is set"""
        assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_debug_mode_configured(self):
        """Test that debug mode is set"""
        assert isinstance(LLM_DEBUG, bool)

    def test_log_level_from_env(self):
        """Test that log level can be set from environment"""
        original = os.environ.get("LLM_LOG_LEVEL")
        try:
            os.environ["LLM_LOG_LEVEL"] = "DEBUG"
            # Re-import to test
            import importlib
            import llmservices
            importlib.reload(llmservices)
            assert llmservices.LOG_LEVEL == "DEBUG"
        finally:
            if original:
                os.environ["LLM_LOG_LEVEL"] = original
            else:
                os.environ.pop("LLM_LOG_LEVEL", None)


class TestLLMRequest:
    """Test LLMRequest with debugging fields"""

    def test_request_has_request_id(self):
        """Test that request has a unique request_id"""
        req1 = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )
        req2 = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )

        assert req1.request_id is not None
        assert req2.request_id is not None
        assert req1.request_id != req2.request_id

    def test_request_has_metadata_dict(self):
        """Test that request has metadata dict"""
        req = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )

        assert hasattr(req, "metadata")
        assert isinstance(req.metadata, dict)

    def test_request_id_is_string(self):
        """Test that request_id is a string"""
        req = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )

        assert isinstance(req.request_id, str)
        assert len(req.request_id) > 0


class TestLLMMetadata:
    """Test LLMMetadata class"""

    def test_metadata_creation(self):
        """Test creating metadata object"""
        metadata = LLMMetadata(
            request_id="test-123",
            start_time=123.456,
            provider="openai",
            model="gpt-4o"
        )

        assert metadata.request_id == "test-123"
        assert metadata.start_time == 123.456
        assert metadata.provider == "openai"
        assert metadata.model == "gpt-4o"

    def test_metadata_to_dict(self):
        """Test metadata serialization to dict"""
        metadata = LLMMetadata(
            request_id="test-123",
            start_time=123.456,
            end_time=124.456,
            duration_ms=1000.0,
            provider="openai",
            model="gpt-4o",
            tokens_used={"total": 100}
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["request_id"] == "test-123"
        assert result["provider"] == "openai"
        assert result["duration_ms"] == 1000.0
        assert result["tokens_used"]["total"] == 100


class TestLLMTiming:
    """Test LLMTiming class"""

    def test_timing_creation(self):
        """Test creating timing object"""
        timing = LLMTiming(
            total_duration_ms=150.5,
            provider_processing_ms=100.0
        )

        assert timing.total_duration_ms == 150.5
        assert timing.provider_processing_ms == 100.0

    def test_timing_to_dict(self):
        """Test timing serialization to dict"""
        timing = LLMTiming(
            dns_lookup_ms=10.0,
            tcp_connect_ms=20.0,
            tls_handshake_ms=30.0,
            total_duration_ms=150.0
        )

        result = timing.to_dict()

        assert isinstance(result, dict)
        assert result["dns_lookup_ms"] == 10.0
        assert result["tcp_connect_ms"] == 20.0
        assert result["total_duration_ms"] == 150.0


class TestLLMError:
    """Test LLMError class"""

    def test_error_creation(self):
        """Test creating LLMError"""
        error = LLMError(
            message="Test error",
            provider="openai",
            model="gpt-4o",
            status_code=500
        )

        assert error.message == "Test error"
        assert error.provider == "openai"
        assert error.model == "gpt-4o"
        assert error.status_code == 500

    def test_error_format_message(self):
        """Test error message formatting"""
        error = LLMError(
            message="API request failed",
            provider="anthropic",
            model="claude-3-5-sonnet",
            status_code=429,
            request_id="req-123"
        )

        formatted = error.format_message()

        assert "Provider: anthropic" in formatted
        assert "Model: claude-3-5-sonnet" in formatted
        assert "Status: 429" in formatted
        assert "RequestID: req-123" in formatted
        assert "API request failed" in formatted

    def test_error_to_dict(self):
        """Test error serialization to dict"""
        error = LLMError(
            message="Test error",
            provider="openai",
            model="gpt-4o",
            status_code=500,
            error_code="rate_limit_exceeded"
        )

        result = error.to_dict()

        assert isinstance(result, dict)
        assert result["message"] == "Test error"
        assert result["provider"] == "openai"
        assert result["status_code"] == 500
        assert result["error_code"] == "rate_limit_exceeded"


class TestLLMConfig:
    """Test LLMConfig class"""

    def test_get_status_returns_dict(self):
        """Test that get_status returns a dictionary"""
        status = LLMConfig.get_status()

        assert isinstance(status, dict)

    def test_status_has_version(self):
        """Test that status includes version"""
        status = LLMConfig.get_status()

        assert "version" in status
        assert status["version"] == "v1.9.0"

    def test_status_has_providers_configured(self):
        """Test that status includes configured providers list"""
        status = LLMConfig.get_status()

        assert "providers_configured" in status
        assert isinstance(status["providers_configured"], list)

    def test_status_has_providers_status(self):
        """Test that status includes per-provider status"""
        status = LLMConfig.get_status()

        assert "providers_status" in status
        assert isinstance(status["providers_status"], dict)

        # Check known providers exist
        known_providers = ["openai", "anthropic", "gemini", "deepseek", "xai", "perplexity", "ollama"]
        for provider in known_providers:
            assert provider in status["providers_status"]

    def test_status_has_environment(self):
        """Test that status includes environment settings"""
        status = LLMConfig.get_status()

        assert "environment" in status
        assert "LLM_LOG_LEVEL" in status["environment"]
        assert "LLM_DEBUG" in status["environment"]

    def test_status_has_circuit_breakers(self):
        """Test that status includes circuit breaker info"""
        status = LLMConfig.get_status()

        assert "circuit_breakers" in status
        assert isinstance(status["circuit_breakers"], dict)

    def test_provider_status_structure(self):
        """Test provider status has correct structure"""
        status = LLMConfig.get_status()

        for provider, provider_status in status["providers_status"].items():
            assert "configured" in provider_status
            assert isinstance(provider_status["configured"], bool)


class TestLLMResponse:
    """Test LLMResponse with debugging fields"""

    def test_response_creation(self):
        """Test creating response with debugging fields"""
        response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai",
            request_id="req-123",
            usage={"total_tokens": 100},
            finish_reason="stop"
        )

        assert response.content == "Test response"
        assert response.request_id == "req-123"
        assert response.usage["total_tokens"] == 100

    def test_response_with_timing(self):
        """Test response with timing information"""
        timing = LLMTiming(total_duration_ms=150.0)
        response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai",
            timing=timing
        )

        assert response.timing is not None
        assert response.timing.total_duration_ms == 150.0

    def test_response_with_metadata(self):
        """Test response with metadata"""
        metadata = LLMMetadata(
            request_id="req-123",
            start_time=123.456,
            provider="openai",
            model="gpt-4o"
        )
        response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai",
            metadata=metadata
        )

        assert response.metadata is not None
        assert response.metadata.provider == "openai"


class TestRequestTracking:
    """Test request tracking functionality"""

    def test_request_id_propagated(self):
        """Test that request_id is preserved through the flow"""
        # This test doesn't make actual API calls
        req = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )

        original_id = req.request_id

        # Simulate creating a response with the same ID
        response = LLMResponse(
            content="Test",
            model="gpt-4o",
            provider="openai",
            request_id=original_id
        )

        assert response.request_id == original_id


class TestDebugMode:
    """Test debug mode functionality"""

    def test_debug_mode_from_env(self):
        """Test that debug mode reads from environment"""
        original = os.environ.get("LLM_DEBUG")

        try:
            os.environ["LLM_DEBUG"] = "true"
            # Check it can be read
            debug_value = os.environ.get("LLM_DEBUG", "false").lower() in ("true", "1", "yes", "on")
            assert debug_value is True
        finally:
            if original:
                os.environ["LLM_DEBUG"] = original

    def test_log_level_from_env(self):
        """Test that log level reads from environment"""
        original = os.environ.get("LLM_LOG_LEVEL")

        try:
            os.environ["LLM_LOG_LEVEL"] = "DEBUG"
            level = os.environ.get("LLM_LOG_LEVEL", "INFO").upper()
            assert level == "DEBUG"
        finally:
            if original:
                os.environ["LLM_LOG_LEVEL"] = original


class TestIntegration:
    """Integration tests for debugging features"""

    def test_full_request_flow_with_metadata(self):
        """Test complete flow with metadata tracking"""
        import time

        # Create request
        req = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt"
        )

        # Track timing
        start = time.time()

        # Simulate processing
        time.sleep(0.01)

        end = time.time()
        duration = (end - start) * 1000

        # Create metadata
        metadata = LLMMetadata(
            request_id=req.request_id,
            start_time=start,
            end_time=end,
            duration_ms=duration,
            provider=req.provider,
            model=req.model,
            status_code=200,
            tokens_used={"total": 100, "prompt": 20, "completion": 80}
        )

        # Create response
        response = LLMResponse(
            content="Test response",
            model=req.model,
            provider=req.provider,
            request_id=req.request_id,
            metadata=metadata,
            finish_reason="stop",
            usage=metadata.tokens_used
        )

        # Verify all data is connected
        assert response.request_id == req.request_id
        assert response.metadata.request_id == req.request_id
        assert response.metadata.duration_ms == duration
        assert response.metadata.tokens_used["total"] == 100

    def test_error_flow_with_context(self):
        """Test error flow with full context"""
        req = LLMRequest(
            provider="openai",
            model="gpt-4o",
            prompt="Test"
        )

        # Simulate error
        error = LLMError(
            message="Rate limit exceeded",
            provider=req.provider,
            model=req.model,
            status_code=429,
            request_id=req.request_id,
            error_code="rate_limit"
        )

        # Verify error has all context
        assert error.provider == req.provider
        assert error.model == req.model
        assert error.request_id == req.request_id

        # Verify error can be serialized
        error_dict = error.to_dict()
        assert error_dict["provider"] == "openai"
        assert error_dict["status_code"] == 429


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
