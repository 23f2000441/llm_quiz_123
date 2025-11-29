import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from quiz_solver import solve_quiz_chain

load_dotenv()

EXPECTED_SECRET = os.getenv("QUIZ_SECRET")
EXPECTED_EMAIL = os.getenv("QUIZ_EMAIL")  # optional check

app = FastAPI(title="LLM Analysis Quiz Endpoint")


# ---------------------------
# Request Body Model
# ---------------------------

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# ---------------------------
# POST /quiz Endpoint
# ---------------------------

@app.post("/quiz")
async def quiz_endpoint(body: QuizRequest):
    """
    Ye endpoint woh hai jo evaluation server hit karega.

    - Invalid JSON  -> automatically handled by Pydantic
    - Secret mismatch -> 403
    - Valid -> 200 + background thread mein quiz solving start
    """

    email = body.email
    secret = body.secret
    url = body.url

    # Optional: Email check
    if EXPECTED_EMAIL and email != EXPECTED_EMAIL:
        print(f"[WARN] Email mismatch: got {email}, expected {EXPECTED_EMAIL}")

    # Secret check
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Background thread so that API returns instantly
    def background_worker():
        print(f"[INFO] Solving quiz URL: {url}")
        start_time = time.time()
        try:
            solve_quiz_chain(
                email=email,
                secret=secret,
                initial_url=url,
                start_time=start_time
            )
        except Exception as e:
            print(f"[ERROR] Quiz solving failed: {repr(e)}")

    t = threading.Thread(target=background_worker, daemon=True)
    t.start()

    # Return immediately
    return JSONResponse(
        status_code=200,
        content={
            "status": "accepted",
            "message": "Quiz solving started in background."
        },
    )
