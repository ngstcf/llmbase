#!/usr/bin/env python3
"""
Test DeepSeek reasoning (thinking) parameter handling.
DeepSeek uses extra_body for the thinking parameter.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("ERROR: DEEPSEEK_API_KEY not found in environment")
    exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def test_without_extra_body():
    """Test current implementation (without extra_body)"""
    print("=" * 60)
    print("Test 1: DeepSeek WITHOUT extra_body (current impl)")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "What is 2+2? Think step by step."}
            ],
            max_tokens=500
        )
        print(f"✓ Success without extra_body")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.choices[0].message.content[:100]}...")

        # Check if reasoning tokens are present
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"  Has reasoning_content: Yes")
        else:
            print(f"  Has reasoning_content: No")

        # Check usage for thinking tokens
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'completion_tokens_details'):
                print(f"  Thinking tokens: {response.usage.completion_tokens_details}")
            else:
                print(f"  Usage: {response.usage}")

        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return False


def test_with_extra_body():
    """Test with extra_body parameter (DeepSeek's recommended approach)"""
    print("=" * 60)
    print("Test 2: DeepSeek WITH extra_body (recommended)")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "What is 2+2? Think step by step."}
            ],
            max_tokens=500,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
        print(f"✓ Success with extra_body")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.choices[0].message.content[:100]}...")

        # Check if reasoning tokens are present
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"  Has reasoning_content: Yes")
            print(f"  Reasoning: {response.choices[0].message.reasoning_content[:100]}...")
        else:
            print(f"  Has reasoning_content: No")

        # Check usage for thinking tokens
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'completion_tokens_details'):
                print(f"  Thinking tokens: {response.usage.completion_tokens_details}")
            else:
                print(f"  Usage: {response.usage}")

        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return False


def test_r1_model():
    """Test DeepSeek-R1 (reasoning model)"""
    print("=" * 60)
    print("Test 3: DeepSeek-R1 (dedicated reasoning model)")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": "What is 2+2? Think step by step."}
            ],
            max_tokens=2000
        )
        print(f"✓ Success with deepseek-reasoner")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.choices[0].message.content[:100]}...")

        # R1 should have reasoning by default
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"  Has reasoning_content: Yes")
            print(f"  Reasoning preview: {response.choices[0].message.reasoning_content[:150]}...")
        else:
            print(f"  Has reasoning_content: No")

        # Check usage for thinking tokens
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'completion_tokens_details'):
                print(f"  Thinking tokens: {response.usage.completion_tokens_details}")
            else:
                print(f"  Total tokens: {response.usage.total_tokens}")

        print()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed: {error_msg}")
        if "404" in error_msg or "not found" in error_msg.lower():
            print("  → Model 'deepseek-reasoner' not available")
        print()
        return False


def main():
    print("\n" + "=" * 60)
    print("Testing DeepSeek Reasoning/Thinking Parameters")
    print("=" * 60 + "\n")

    # Run tests
    test_without_extra_body()
    test_with_extra_body()
    test_r1_model()

    print("=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    print("""
Current Implementation Status:
  1. Without extra_body: [Check test result above]
  2. With extra_body:    [Check test result above]
  3. R1 model:          [Check test result above]

If Test 1 works:
  → Current llmservices.py code works for basic DeepSeek
  → Thinking may be handled automatically by DeepSeek

If Test 2 works AND provides reasoning_content:
  → Need to update DeepSeekProvider to pass extra_body
  → Modify OpenAIProvider.call() to accept extra_body parameter

If Test 3 works with reasoning_content:
  → R1 model includes reasoning by default
  → No special parameters needed
    """)


if __name__ == "__main__":
    main()
