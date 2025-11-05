# tests/test_generation.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_generate_simple():
    prompt = "Write a Python function that returns the factorial of n.\n\n### Solution\n"
    resp = requests.post(BASE_URL + "/generate", json={"prompt": prompt, "max_new_tokens": 200})
    assert resp.status_code == 200
    completion = resp.json()["completion"]
    assert "def" in completion and "return" in completion
