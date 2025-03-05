import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file!")

print("API Key Loaded Successfully:", api_key[:5] + "********")  # Mask for security
