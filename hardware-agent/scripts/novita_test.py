import os
from openai import OpenAI

api_key = os.getenv("NOVITA_API_KEY")
if not api_key:
    raise ValueError("NOVITA_API_KEY environment variable is not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.novita.ai/openai"
)

response = client.chat.completions.create(
    model="google/gemma-4-31b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=131072,
    temperature=0.7
)

print(response.choices[0].message.content)
