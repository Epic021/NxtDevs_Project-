from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from app.rag import RAGPipeline  # Fix import path
from app.user_profile import update_user_profile
from app.config import estimate_user_level, save_assessment_mcqs, diverse_mcq_query
from typing import List

# Load environment variables from the .env file
load_dotenv()

# Access environment variables for API keys
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the RAG model with API key
rag_model = RAGPipeline(api_key=google_api_key)

# MODELS
class QuestionRequest(BaseModel):
    topic: str
    subtopic: str
    user_level: int

class AutoAssessRequest(BaseModel):
    topic: str
    subtopic: str

class SubmitAssessmentRequest(BaseModel):
    topic: str
    subtopic: str
    responses: List[dict]  # [{question, selected, correct}]

# MANUAL MCQ GENERATION
@app.post("/generate_mcq/")
async def generate_mcq(request: QuestionRequest):
    query = diverse_mcq_query(request.topic, request.subtopic, request.user_level)
    try:
        mcq = rag_model.generate_mcq(request.topic.lower(), request.subtopic.lower(), query)
        if not mcq:
            raise ValueError("Failed to generate MCQ")
        update_user_profile(request.topic, request.subtopic, request.user_level, mcq)
        return {"mcq": mcq}
    except Exception as e:
        # Improved logging
        print(f"Error in /generate_mcq/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# AUTO ASSESSMENT - MCQ GENERATION
@app.post("/auto_assess/")
async def auto_assess(request: AutoAssessRequest):
    mcqs = []
    for level in range(1, 4):
        query = diverse_mcq_query(request.topic, request.subtopic, level)
        mcq = rag_model.generate_mcq(request.topic.lower(), request.subtopic.lower(), query)
        mcqs.append({"level": level, "mcq": mcq})
    save_assessment_mcqs(request.topic, request.subtopic, mcqs)
    return {"questions": mcqs}

# AUTO ASSESSMENT - USER SUBMITS ANSWERS
@app.post("/submit_assessment/")  
async def submit_assessment(data: SubmitAssessmentRequest):
    estimated_level = estimate_user_level(data.responses)
    update_user_profile(data.topic, data.subtopic, estimated_level)
    return {"estimated_level": estimated_level}

# Optional: Manual update
@app.post("/update_profile/")
async def update_profile(topic: str, subtopic: str, level: int):
    update_user_profile(topic, subtopic, level)
    return {"message": "User profile updated successfully"}

@app.get("/test_api/")
async def test_api():
    try:
        result = rag_model.test_api_connection()
        return {"status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
