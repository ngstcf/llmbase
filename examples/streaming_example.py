"""
Streaming Example for LLM Services

This example demonstrates how to stream responses from LLM providers.
Streaming is useful for long-form content generation and real-time applications.

Prerequisites:
1. Install dependencies: pip install -r requirements.txt
2. Set up your .env file with API keys
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmservices import LLMService, LLMRequest, CircuitBreakerOpenException
from dotenv import load_dotenv


def parse_stream_chunk(chunk: str) -> str:
    """
    Parse SSE (Server-Sent Events) format and extract the text content.

    Handles formats like:
    - data: {"chunk": "text"}
    - data: {"delta": {"content": "text"}}
    - data: [DONE] (stream end marker)
    - raw text

    Returns:
        str: The extracted text content, or empty string for [DONE] markers
    """
    chunk = chunk.strip()

    # Handle SSE format: data: {...}
    if chunk.startswith('data: '):
        data_part = chunk[6:]  # Remove "data: " prefix

        # Check for stream end marker
        if data_part.strip() == '[DONE]':
            return ''  # Signal end of stream

        try:
            data = json.loads(data_part)

            # Try different possible keys for the content
            if 'chunk' in data:
                return data['chunk']
            elif 'delta' in data and 'content' in data['delta']:
                return data['delta']['content']
            elif 'content' in data:
                return data['content']
            elif 'text' in data:
                return data['text']
        except json.JSONDecodeError:
            pass

    # Return as-is if not SSE format or parsing failed
    return chunk


def stream_basic_example():
    """Basic streaming example with default settings."""
    print("=" * 60)
    print("Example 1: Basic Streaming")
    print("=" * 60)

    llm_request = LLMRequest(
        provider='openai',
        model='gpt-4o',
        prompt='Write a short haiku about artificial intelligence.',
        stream=True
    )

    try:
        print("\nStreaming response: ")
        for raw_chunk in LLMService.stream(llm_request):
            text = parse_stream_chunk(raw_chunk)
            print(text, end='', flush=True)
        print("\n✓ Stream complete")
    except CircuitBreakerOpenException as e:
        print(f"\n✗ Service Unavailable (Circuit Breaker): {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def stream_with_accumulation_example():
    """Example showing how to accumulate streamed content."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming with Content Accumulation")
    print("=" * 60)

    llm_request = LLMRequest(
        provider='anthropic',
        model='claude-sonnet-4-5-20250929',
        prompt='List 3 benefits of using Python for AI development.',
        stream=True
    )

    full_response = ""
    chunk_count = 0

    try:
        print("\nStreaming response: ")
        for raw_chunk in LLMService.stream(llm_request):
            text = parse_stream_chunk(raw_chunk)
            if text:  # Skip empty strings (filters out [DONE])
                print(text, end='', flush=True)
                full_response += text
                chunk_count += 1

        print(f"\n\n✓ Stream complete")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total characters: {len(full_response)}")
    except CircuitBreakerOpenException as e:
        print(f"\n✗ Service Unavailable (Circuit Breaker): {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def stream_json_mode_example():
    """Example of streaming with JSON mode enabled."""
    print("\n" + "=" * 60)
    print("Example 3: Streaming with JSON Mode")
    print("=" * 60)

    llm_request = LLMRequest(
        provider='openai',
        model='gpt-4o',
        prompt='List 3 programming languages with their release years in JSON format.',
        json_mode=True,
        stream=True
    )

    full_response = ""

    try:
        print("\nStreaming JSON response: ")
        for raw_chunk in LLMService.stream(llm_request):
            text = parse_stream_chunk(raw_chunk)
            if text:
                print(text, end='', flush=True)
                full_response += text

        # Parse the accumulated JSON
        print("\n\n✓ Stream complete, parsing JSON...")
        try:
            data = json.loads(full_response)
            print(f"\nParsed data:\n{json.dumps(data, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"\n✗ Failed to parse JSON: {e}")
    except CircuitBreakerOpenException as e:
        print(f"\n✗ Service Unavailable (Circuit Breaker): {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def stream_different_providers_example():
    """Example showing streaming from different providers."""
    print("\n" + "=" * 60)
    print("Example 4: Streaming from Different Providers")
    print("=" * 60)

    providers = [
        ("openai", "gpt-4o", "OpenAI GPT-4o"),
        ("anthropic", "claude-sonnet-4-5-20250929", "Anthropic Claude"),
        ("gemini", "gemini-2.5-flash-exp", "Google Gemini"),
    ]

    for provider, model, name in providers:
        print(f"\n--- Testing {name} ---")

        llm_request = LLMRequest(
            provider=provider,
            model=model,
            prompt='Say "Hello from [provider]!" in one sentence.',
            stream=True
        )

        try:
            for raw_chunk in LLMService.stream(llm_request):
                text = parse_stream_chunk(raw_chunk)
                print(text, end='', flush=True)
            print()  # New line after each provider
        except Exception as e:
            print(f"✗ Failed: {e}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("\n" + "🌊 LLM Services - Streaming Examples" + "\n")

    # Run examples
    stream_basic_example()
    stream_with_accumulation_example()
    stream_json_mode_example()
    stream_different_providers_example()

    print("\n" + "=" * 60)
    print("All streaming examples completed!")
    print("=" * 60 + "\n")
