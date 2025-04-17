import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Load your JSON file
with open("data.json", "r") as f:
    json_data = json.load(f)

# Convert JSON to string
json_str = json.dumps(json_data, indent=2)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-8b')

# Instructional prompt for Gemini
prompt = f"""
You are an interview analysis expert.

when i give you behavioral data from a user's interview (JSON format). 
Ignore any code inconsistencies—assume the data is accurate.

Your task will be to :
- Identify key behavior traits (e.g., filler words, clarity, pace, pauses).
- Provide brief, actionable insights to help the user improve in future interviews.

"""

chat = model.start_chat(
    history=[
        {"role": "user", "parts": prompt},
        {"role": "model", "parts": "Okay, I'm ready.  Please provide the JSON data from the user's interview."},
    ]
)

output = chat.send_message(f" Json Data : {json_str}")

Improvements = output.text
print(Improvements)