#!/usr/bin/env python3
"""
Test script to verify reasoning_effort parameter handling for current and future models.
This tests both the current chat.completions format and the newer Responses API format.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def test_current_chat_completions():
    """Test current format using chat.completions with reasoning_effort (for o1)"""
    print("=" * 60)
    print("Test 1: Current chat.completions API with reasoning_effort")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="gpt-5",  # or o1-mini
            messages=[
                {"role": "user", "content": "What is 2+2? Think step by step."}
            ],
            max_completion_tokens=1000,
            reasoning_effort="low"
        )
        print(f"✓ Success with chat.completions + reasoning_effort")
        print(f"  Model: {response.model}")
        print(f"  Finish reason: {response.choices[0].finish_reason}")
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"  Has reasoning_content: Yes")
        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return False


def test_new_responses_api():
    """Test newer Responses API format (for gpt-5 and future models)"""
    print("=" * 60)
    print("Test 2: New Responses API with reasoning object")
    print("=" * 60)

    try:
        # This is the newer API format you found in documentation
        response = client.responses.create(
            model="gpt-5",
            input="What is the capital of France?",
            reasoning={
                "effort": "low",
                "summary": "auto"
            }
        )
        print(f"✓ Success with responses.create + reasoning object")
        print(f"  Output: {response.output}")
        print()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed: {error_msg}")

        # Parse specific errors
        if "404" in error_msg or "not found" in error_msg.lower():
            print("  → Model 'gpt-5' not available (expected - not yet public)")
        elif "method" in error_msg.lower() or "attribute" in error_msg.lower():
            print("  → client.responses.create() not available in this SDK version")
        else:
            print(f"  → Other error: {error_msg}")
        print()
        return False


def test_parameter_format():
    """Test what parameter format works"""
    print("=" * 60)
    print("Test 3: Check parameter format compatibility")
    print("=" * 60)

    # Test with o1 using current format
    try:
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=100,
            reasoning_effort="low"
        )
        print("✓ o1-mini accepts: reasoning_effort='low' (string)")
    except Exception as e:
        print(f"✗ reasoning_effort as string failed: {e}")

    # Try with dict format (unlikely to work)
    try:
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=100,
            reasoning={"effort": "low"}  # Try nested format
        )
        print("✓ o1-mini accepts: reasoning={'effort': 'low'} (dict)")
    except Exception as e:
        print(f"✗ reasoning as dict failed (expected for chat.completions)")
    print()


def main():
    print("\n" + "=" * 60)
    print("Testing Reasoning Parameter Formats")
    print("=" * 60 + "\n")

    # Run tests
    test_current_chat_completions()
    test_new_responses_api()
    test_parameter_format()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
For CURRENT implementation (llmservices.py):
  - Uses: client.chat.completions.create()
  - Parameter: reasoning_effort="low" (flat string)
  - Works with: o1, o1-mini, o3 models

For FUTURE implementation (when gpt-5 is available):
  - Will use: client.responses.create()
  - Parameter: reasoning={"effort": "low", "summary": "auto"}
  - Works with: gpt-5 and future models

Action required: Update code to detect model type and use appropriate endpoint.
    """)


if __name__ == "__main__":
    main()
