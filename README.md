# Personal Agent API

This is a FastAPI app that publishes a chatbot endpoint acting as **your agent**.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key_here
uvicorn app.main:app --reload
