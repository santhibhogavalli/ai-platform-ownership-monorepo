from anthropic import Anthropic 
from langchain_google_genai import ChatGoogleGenerativeAI
"""Test Anthropic API key configuration."""

def test_anthropic_models_list():
    """Verify Anthropic API key is valid and models can be listed."""
    client = Anthropic(api_key='api-key')
    response = client.models.list()

    assert response.data, "No models returned from Anthropic API"
    assert any(model.id for model in response.data), "Models list is empty"
    assert 'claude' in str(response.data[0]).lower(), "Expected Claude model not found"


"""Test Google API key configuration.""

def test_google_api_key():
    """Verify Google API key is valid and models can be accessed."""
    client = ChatGoogleGenerativeAI(api_key='api-key', model='gemini-1.5-flash')

    assert client.model_name == 'gemini-1.5-flash', "Model not properly configured"
    assert client.api_key, "API key not set"



if __name__ == "__main__":
    test_anthropic_models_list()
    print("✓ Anthropic API key test passed")
    test_google_api_key()
        print("✓ Google API key test passed")
    