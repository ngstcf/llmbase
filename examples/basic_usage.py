"""
Basic Usage Example for LLM Services

This example demonstrates the simplest way to use LLMService as a Python library.
No Flask required - just import and use.

Prerequisites:
1. Install dependencies: pip install -r requirements.txt
2. Set up your .env file with API keys
3. (Optional) Create llm_config.json with custom model configurations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmservices import LLMService, LLMRequest


def simple_call_example():
    """Basic example of making a simple LLM call."""
    print("=" * 60)
    print("Example 1: Simple LLM Call")
    print("=" * 60)

    req = LLMRequest(
        provider="openai",
        model="gpt-4o",
        prompt="Write a haiku about artificial intelligence"
    )

    try:
        response = LLMService.call(req)
        print(f"\nProvider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Response:\n{response.content}")
        if response.usage:
            print(f"\nTokens used: {response.usage.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")


def different_providers_example():
    """Example of using different providers."""
    print("\n" + "=" * 60)
    print("Example 2: Different Providers")
    print("=" * 60)

    providers = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("gemini", "gemini-2.5-flash-exp"),
    ]

    for provider, model in providers:
        print(f"\nTrying {provider}/{model}...")
        req = LLMRequest(
            provider=provider,
            model=model,
            prompt="Say 'Hello from [provider]!' in one sentence."
        )

        try:
            response = LLMService.call(req)
            print(f"✓ {response.content}")
        except Exception as e:
            print(f"✗ Failed: {e}")


def parameters_example():
    """Example of using various parameters."""
    print("\n" + "=" * 60)
    print("Example 3: Using Parameters")
    print("=" * 60)

    req = LLMRequest(
        provider="openai",
        model="gpt-4o",
        prompt="Write a very short creative story about a robot",
        temperature=0.9,      # Higher creativity
        max_tokens=100,       # Limit response length
        system_prompt="You are a creative writer specializing in science fiction."
    )

    try:
        response = LLMService.call(req)
        print(f"\nResponse:\n{response.content}")
    except Exception as e:
        print(f"Error: {e}")


def chat_example():
    """Example of using chat messages instead of a simple prompt."""
    print("\n" + "=" * 60)
    print("Example 4: Chat with Messages")
    print("=" * 60)

    req = LLMRequest(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        messages=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
            {"role": "user", "content": "What are its main features?"}
        ]
    )

    try:
        response = LLMService.call(req)
        print(f"\nResponse:\n{response.content}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "🚀 LLM Services - Basic Usage Examples" + "\n")

    # Run examples
    simple_call_example()
    different_providers_example()
    parameters_example()
    chat_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")
