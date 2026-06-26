from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def load_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found in environment variables")
        return None
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client
