# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "interview_data")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Load from .env

# Add other configuration variables here