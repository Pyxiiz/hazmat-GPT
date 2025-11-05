# server.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

app = FastAPI(title="Mini-GPT Code Generator")

MODEL_DIR = os.environ.get("MODEL_DIR", "out-lora")  # path to fine-tuned model

@app.on_event("startup")
async def load_model():
    global tokenizer, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    print("Model loaded on", device)

@app.post("/generate")
async def generate(req: GenRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=False,
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the completion part after the prompt
    completion = text[len(req.prompt):].lstrip()
    return {"completion": completion}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
