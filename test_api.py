import os
import sys

# Try to import openai, handle if not installed (though it should be)
try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' library not installed.")
    sys.exit(1)

# Check for API Key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # Try loading from .env manually if not in env vars (for local testing convenience)
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break
    except FileNotFoundError:
        pass

if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
    sys.exit(1)

print(f"Key found: {api_key[:8]}...{api_key[-4:]}")

client = OpenAI(api_key=api_key)

try:
    print("Sending request to model: gpt-4o-mini...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say Hello"}],
        max_tokens=10
    )
    print("\nSuccess! The API is working.")
    print("-" * 20)
    print("Response from LLM:", response.choices[0].message.content)
    print("-" * 20)

except Exception as e:
    print("\nAPI Test Failed!")
    print(e)
