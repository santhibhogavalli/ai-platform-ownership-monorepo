import os
from dotenv import load_dotenv
from services.gateway.app.providers.registry import ProviderRegistry


def main():
    load_dotenv()

    registry = ProviderRegistry()
    print("Available providers:", list(registry.providers.keys()))
    print("Available models:")
    for model in registry.list_all_models():
        print(f"  {model['provider']}: {model['name']} (${model['cost_per_input_token']*1000:.3f}/1k input tokens)")

    if registry.providers:
        print("\nTesting complete_with_fallback()...")
        try:
            result = registry.complete_with_fallback("What is the capital of India?", max_tokens=50)
            print(f"Response: {result['text'][:100]}...")
            print(f"Model: {result['model']['name']} ({result['model']['provider']})")
            print(f"Tokens: {result['tokens']['input']} input, {result['tokens']['output']} output")
            print(f"Cost: ${result['cost']['total_cost']:.6f}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No providers configured. Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_PRIMARY_URL.")


if __name__ == "__main__":
    main()
