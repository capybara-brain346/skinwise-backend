from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

def load_gemini():
    global gemini_client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        return None
    gemini_client = genai.Client(api_key=api_key)
    return gemini_client