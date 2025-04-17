import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (ensure your GOOGLE_API_KEY is in a .env file)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
else:
    genai.configure(api_key=api_key)

    print("Available models:")
    # List models available for the 'generateContent' method
    for m in genai.list_models():
      if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")

    print("\nAvailable embedding models:")
    # List models available for the 'embedContent' method
    for m in genai.list_models():
      if 'embedContent' in m.supported_generation_methods:
        print(f"- {m.name}")
