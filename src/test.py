from anthropic import Anthropic
client = Anthropic(api_key='api-key')

print("\n📋 Your Available Models:\n")
response = client.models.list()

for model in response.data:
    print(f"  ✅ {model.id}")
    