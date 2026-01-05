"""
Quick LLM Model Tester
Interactive testing tool for individual models
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from colorama import init, Fore, Style

init(autoreset=True)
load_dotenv()

API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "http://localhost:8888")


def print_header(text):
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{text.center(80)}")
    print(f"{Fore.CYAN}{'='*80}\n")


def print_response(data):
    """Pretty print API response"""
    print(f"\n{Fore.GREEN}Response:")
    print(f"{Fore.WHITE}{'-'*80}")
    print(data.get('content', 'No content'))
    
    if data.get('reasoning_content'):
        print(f"\n{Fore.YELLOW}Reasoning:")
        print(f"{Fore.WHITE}{'-'*80}")
        print(data['reasoning_content'])
    
    if data.get('usage'):
        print(f"\n{Fore.CYAN}Token Usage:")
        usage = data['usage']
        print(f"  Prompt: {usage.get('prompt_tokens', 0)}")
        print(f"  Completion: {usage.get('completion_tokens', 0)}")
        print(f"  Total: {usage.get('total_tokens', 0)}")
    
    if data.get('finish_reason'):
        print(f"\n{Fore.MAGENTA}Finish Reason: {data['finish_reason']}")


def test_model_interactive():
    """Interactive model testing"""
    print_header("Quick LLM Model Tester")
    
    # Get provider
    print(f"{Fore.YELLOW}Available providers:")
    print("1. openai")
    print("2. anthropic")
    print("3. gemini")
    print("4. deepseek")
    print("5. xai (Grok)")
    print("6. perplexity")
    print("7. azure_oai")
    print("8. ollama")

    provider_map = {
        '1': 'openai',
        '2': 'anthropic',
        '3': 'gemini',
        '4': 'deepseek',
        '5': 'xai',
        '6': 'perplexity',
        '7': 'azure_oai',
        '8': 'ollama'
    }

    choice = input(f"\n{Fore.GREEN}Select provider (1-8): ").strip()
    provider = provider_map.get(choice)
    
    if not provider:
        print(f"{Fore.RED}Invalid choice")
        return
    
    # Get available models
    try:
        response = requests.get(f"{API_BASE_URL}/api/providers/{provider}/models")
        response.raise_for_status()
        models_data = response.json()
        models = models_data.get('models', [])
        
        if not models:
            print(f"{Fore.RED}No models available for {provider}")
            return
        
        print(f"\n{Fore.YELLOW}Available models for {provider}:")
        for i, model in enumerate(models, 1):
            if isinstance(model, dict):
                name = model['name']
                flags = []
                if model.get('supports_streaming'):
                    flags.append('streaming')
                if model.get('supports_reasoning'):
                    flags.append('reasoning')
                if model.get('supports_extended_thinking'):
                    flags.append('thinking')
                flags_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"{i}. {name}{flags_str}")
            else:
                print(f"{i}. {model}")
        
        model_idx = int(input(f"\n{Fore.GREEN}Select model number: ")) - 1
        if model_idx < 0 or model_idx >= len(models):
            print(f"{Fore.RED}Invalid model number")
            return
        
        model = models[model_idx]
        if isinstance(model, dict):
            model_name = model['name']
            supports_reasoning = model.get('supports_reasoning', False)
        else:
            model_name = model
            supports_reasoning = False
        
    except Exception as e:
        print(f"{Fore.RED}Error fetching models: {e}")
        return
    
    # Get prompt
    print(f"\n{Fore.YELLOW}Enter your prompt (or press Enter for default):")
    prompt = input().strip()
    if not prompt:
        prompt = "What is 2+2? Explain briefly."
    
    # Get system prompt
    print(f"\n{Fore.YELLOW}Enter system prompt (optional, press Enter to skip):")
    system_prompt = input().strip() or None
    
    # Configuration
    print(f"\n{Fore.YELLOW}Configuration:")
    stream = input("Enable streaming? (y/n, default: n): ").lower() == 'y'
    
    max_tokens = input("Max tokens (default: 500): ").strip()
    max_tokens = int(max_tokens) if max_tokens else 500
    
    temperature = input("Temperature (default: 0.7): ").strip()
    temperature = float(temperature) if temperature else 0.7
    
    enable_thinking = False
    reasoning_effort = None

    if supports_reasoning:
        enable_thinking = input("Enable thinking/reasoning? (y/n, default: y): ").lower() != 'n'
        if enable_thinking:
            # OpenAI supports reasoning_effort for o1, gpt-5 models
            if provider in ['openai', 'azure_oai'] and any(m in model_name for m in ['o1', 'gpt-5']):
                print("Reasoning effort (low/medium/high, default: medium):")
                reasoning_effort = input().strip() or "medium"
            # DeepSeek does not support effort levels, only on/off
            elif provider == 'deepseek':
                print(f"{Fore.CYAN}Note: DeepSeek thinking is either on or off (no effort levels)")
            # xAI Grok 4 does not support effort levels (reasoning is automatic)
            elif provider == 'xai' and 'grok-4' in model_name:
                print(f"{Fore.CYAN}Note: Grok 4 reasoning is automatic (no effort levels)")
            # Anthropic does not use reasoning_effort parameter
            elif provider == 'anthropic':
                print(f"{Fore.CYAN}Note: Anthropic extended thinking is automatic")
    
    # Build request
    payload = {
        "provider": provider,
        "model": model_name,
        "prompt": prompt,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking
    }
    
    if system_prompt:
        payload["system_prompt"] = system_prompt
    
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    
    # Make request
    print(f"\n{Fore.CYAN}Testing {provider}/{model_name}...")
    print(f"{Fore.WHITE}{'-'*80}\n")
    
    try:
        if stream:
            response = requests.post(
                f"{API_BASE_URL}/api/llm/call",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            print(f"{Fore.GREEN}Streaming response:\n")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            print(f"\n{Fore.CYAN}[Stream complete]")
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'chunk' in chunk:
                                print(chunk['chunk'], end='', flush=True)
                            elif 'reasoning' in chunk:
                                reasoning_text = chunk['reasoning']
                                # Only print header once
                                if not hasattr(locals(), 'reasoning_header_shown'):
                                    print(f"\n{Fore.YELLOW}Reasoning:", end='')
                                    reasoning_header_shown = True

                                print(f" {reasoning_text}", end='', flush=True)
                            elif 'error' in chunk:
                                print(f"\n{Fore.RED}Error: {chunk['error']}")
                        except json.JSONDecodeError:
                            continue
        else:
            response = requests.post(
                f"{API_BASE_URL}/api/llm/call",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                print(f"{Fore.RED}Error: {data['error']}")
            else:
                print_response(data)
    
    except requests.exceptions.Timeout:
        print(f"{Fore.RED}Request timed out")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")


def test_specific_model(provider, model, prompt):
    """Test a specific model with a prompt"""
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "stream": False,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        print(f"\n{Fore.CYAN}Testing {provider}/{model}...")
        response = requests.post(
            f"{API_BASE_URL}/api/llm/call",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"{Fore.RED}Error: {data['error']}")
        else:
            print_response(data)
    
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 3:
        # Command line mode: python quick_test.py <provider> <model> <prompt>
        provider = sys.argv[1]
        model = sys.argv[2]
        prompt = ' '.join(sys.argv[3:])
        test_specific_model(provider, model, prompt)
    else:
        # Interactive mode
        test_model_interactive()


if __name__ == "__main__":
    main()