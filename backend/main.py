import os
import time
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    prompt: str
    system_instruction: str = "You are a helpful AI assistant."
    groq_api_key: Optional[str] = None
    groq_small_model: str = "llama-3.1-8b-instant"
    groq_big_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "llama3.2"


class ModelResult(BaseModel):
    model: str
    text: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class EvaluateResponse(BaseModel):
    groq_small: ModelResult
    groq_big: ModelResult
    ollama: ModelResult


app = FastAPI(title="DevPrompt Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "DevPrompt Lab API"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    groq_key = req.groq_api_key or os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise HTTPException(status_code=400, detail="Groq API key is required.")

    # Groq small model call
    groq_client = Groq(api_key=groq_key)
    small_start = time.perf_counter()
    small_resp = groq_client.chat.completions.create(
        model=req.groq_small_model,
        messages=[
            {"role": "system", "content": req.system_instruction},
            {"role": "user", "content": req.prompt},
        ],
    )
    small_latency = (time.perf_counter() - small_start) * 1000

    small_choice = small_resp.choices[0].message
    small_usage = getattr(small_resp, "usage", None)

    groq_small_result = ModelResult(
        model=req.groq_small_model,
        text=small_choice.content if isinstance(small_choice.content, str) else str(small_choice.content),
        latency_ms=small_latency,
        input_tokens=getattr(small_usage, "prompt_tokens", None) if small_usage else None,
        output_tokens=getattr(small_usage, "completion_tokens", None) if small_usage else None,
        total_tokens=getattr(small_usage, "total_tokens", None) if small_usage else None,
    )

    # Groq big model call
    big_start = time.perf_counter()
    big_resp = groq_client.chat.completions.create(
        model=req.groq_big_model,
        messages=[
            {"role": "system", "content": req.system_instruction},
            {"role": "user", "content": req.prompt},
        ],
    )
    big_latency = (time.perf_counter() - big_start) * 1000

    big_choice = big_resp.choices[0].message
    big_usage = getattr(big_resp, "usage", None)

    groq_big_result = ModelResult(
        model=req.groq_big_model,
        text=big_choice.content if isinstance(big_choice.content, str) else str(big_choice.content),
        latency_ms=big_latency,
        input_tokens=getattr(big_usage, "prompt_tokens", None) if big_usage else None,
        output_tokens=getattr(big_usage, "completion_tokens", None) if big_usage else None,
        total_tokens=getattr(big_usage, "total_tokens", None) if big_usage else None,
    )

    # Ollama (local) call via OpenAI-compatible endpoint
    o_start = time.perf_counter()
    try:
        o_resp = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": req.ollama_model,
                "messages": [
                    {"role": "system", "content": req.system_instruction},
                    {"role": "user", "content": req.prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        o_resp.raise_for_status()
        o_json = o_resp.json()
        o_text = o_json.get("message", {}).get("content", "")
    except Exception as e:
        o_text = f"Ollama request failed: {e}"
    o_latency = (time.perf_counter() - o_start) * 1000

    ollama_result = ModelResult(
        model=req.ollama_model,
        text=o_text,
        latency_ms=o_latency,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
    )

    return EvaluateResponse(groq_small=groq_small_result, groq_big=groq_big_result, ollama=ollama_result)

