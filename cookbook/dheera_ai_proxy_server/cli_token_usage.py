#!/usr/bin/env python3
"""
Example: Using CLI token with DheeraAI SDK

This example shows how to use the CLI authentication token
in your Python scripts after running `dheera_ai-proxy login`.
"""

from textwrap import indent
import dheera_ai
DHEERA_AI_BASE_URL = "http://localhost:4000/"


def main():
    """Using CLI token with DheeraAI SDK"""
    print("🚀 Using CLI Token with DheeraAI SDK")
    print("=" * 40)
    #dheera_ai._turn_on_debug()
    
    # Get the CLI token
    api_key = dheera_ai.get_dheera_ai_gateway_api_key()
    
    if not api_key:
        print("❌ No CLI token found. Please run 'dheera_ai-proxy login' first.")
        return
    
    print("✅ Found CLI token.")

    available_models = dheera_ai.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider="dheera_ai_proxy",
        api_key=api_key,
        api_base=DHEERA_AI_BASE_URL
    )
    
    print("✅ Available models:")
    if available_models:
        for i, model in enumerate(available_models, 1):
            print(f"   {i:2d}. {model}")
    else:
        print("   No models available")
    
    # Use with DheeraAI
    try:
        response = dheera_ai.completion(
            model="dheera_ai_proxy/gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello from CLI token!"}],
            api_key=api_key,
            base_url=DHEERA_AI_BASE_URL
        )
        print(f"✅ LLM Response: {response.model_dump_json(indent=4)}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
    
    print("\n💡 Tips:")
    print("1. Run 'dheera_ai-proxy login' to authenticate first")
    print("2. Replace 'https://your-proxy.com' with your actual proxy URL")
    print("3. The token is stored locally at ~/.dheera_ai/token.json")
