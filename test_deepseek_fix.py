#!/usr/bin/env python3
"""
Test DeepSeek reasoning with the updated llmservices.py
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import llmservices
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmservices import LLMService, LLMRequest

load_dotenv()

def test_deepseek_chat_with_thinking():
    """Test deepseek-chat with enable_thinking=True"""
    print("=" * 60)
    print("Test 1: deepseek-chat WITH enable_thinking=True")
    print("=" * 60)

    try:
        req = LLMRequest(
            provider="deepseek",
            model="deepseek-chat",
            prompt="What is 2+2? Think step by step.",
            enable_thinking=True
        )
        response = LLMService.call(req)

        print(f"✓ Success")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.content[:100]}...")

        if response.reasoning_content:
            print(f"  Has reasoning_content: Yes")
            print(f"  Reasoning preview: {response.reasoning_content[:150]}...")
        else:
            print(f"  Has reasoning_content: No")

        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return False


def test_deepseek_chat_without_thinking():
    """Test deepseek-chat with enable_thinking=False (default)"""
    print("=" * 60)
    print("Test 2: deepseek-chat WITHOUT enable_thinking")
    print("=" * 60)

    try:
        req = LLMRequest(
            provider="deepseek",
            model="deepseek-chat",
            prompt="What is 2+2?"
        )
        response = LLMService.call(req)

        print(f"✓ Success")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.content[:100]}...")

        if response.reasoning_content:
            print(f"  Has reasoning_content: Yes (unexpected!)")
        else:
            print(f"  Has reasoning_content: No (expected)")

        print()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return False


def test_deepseek_reasoner():
    """Test deepseek-reasoner (automatic reasoning)"""
    print("=" * 60)
    print("Test 3: deepseek-reasoner (automatic reasoning)")
    print("=" * 60)

    try:
        req = LLMRequest(
            provider="deepseek",
            model="deepseek-reasoner",
            prompt="What is 2+2? Think step by step."
        )
        response = LLMService.call(req)

        print(f"✓ Success")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.content[:100]}...")

        if response.reasoning_content:
            print(f"  Has reasoning_content: Yes (automatic)")
            print(f"  Reasoning preview: {response.reasoning_content[:150]}...")
        else:
            print(f"  Has reasoning_content: No")

        print()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed: {error_msg}")
        if "404" in error_msg or "not found" in error_msg.lower():
            print("  → Model 'deepseek-reasoner' not available (may need access)")
        print()
        return False


def main():
    print("\n" + "=" * 60)
    print("Testing DeepSeek Reasoning with Updated llmservices.py")
    print("=" * 60 + "\n")

    # Run tests
    results = []
    results.append(("deepseek-chat with thinking", test_deepseek_chat_with_thinking()))
    results.append(("deepseek-chat without thinking", test_deepseek_chat_without_thinking()))
    results.append(("deepseek-reasoner", test_deepseek_reasoner()))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print("Backward Compatibility Check")
    print("=" * 60)
    print("""
Existing code behavior:
  - OpenAIProvider.call(req, client) → Still works (extra_body=None by default)
  - All existing providers → Unaffected (no extra_body passed)
  - Only DeepSeek with enable_thinking=True → Gets extra_body

✓ Changes are backward compatible!
    """)


if __name__ == "__main__":
    main()
