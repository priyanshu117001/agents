from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import openai
import os


from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
from app.agent import Me
from fastapi.middleware.cors import CORSMiddleware 



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 For testing; replace with ["https://yourwebsite.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow Content-Type, Authorization, etc.
)
# Initialize FastAPI
app = FastAPI(title="Personal Agent API", version="1.0")

# Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define request/response schema
class AgentRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    reply: str

# Root endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "Your agent is running 🚀"}

# Chat endpoint
@app.post("/chat")
async def chat(req: AgentRequest):
    """
    Endpoint for interacting with your agent.
    """
    try:
        chat = Me()
        reply = chat.chat(req.message)
        return {"response": reply}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}

# Run locally
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
