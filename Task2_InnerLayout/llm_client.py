import json
import urllib.request
import urllib.error
import sys

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    OPENAI_API_BASE
)

def chat(model, messages, temperature=None, response_format=None):
    provider = LLM_PROVIDER.strip().lower()

    if provider == "openai":
        if not OPENAI_API_KEY:
            print("\n[ERROR] OPENAI_API_KEY is not set in the environment or .env file.")
            sys.exit(1)
        
        url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096
        }
        if "deepseek" in model.lower():
            data["chat_template_kwargs"] = {
                "thinking": False
            }
        if temperature is not None:
            data["temperature"] = temperature
        if response_format is not None:
            data["response_format"] = response_format

    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            print("\n[ERROR] ANTHROPIC_API_KEY is not set in the environment or .env file.")
            sys.exit(1)
            
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096
        }
        if temperature is not None:
            data["temperature"] = temperature

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            print("\n[ERROR] GEMINI_API_KEY is not set in the environment or .env file.")
            sys.exit(1)
            
        clean_model = model
        if not clean_model.startswith("models/"):
            clean_model = f"models/{clean_model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={GEMINI_API_KEY}"
        headers = {
            "Content-Type": "application/json"
        }
        
        contents = []
        for m in messages:
            role = "user" if m.get("role") == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m.get("content", "")}]
            })
            
        data = {
            "contents": contents
        }
        if temperature is not None:
            data["generationConfig"] = {
                "temperature": temperature
            }

    else:
        print(f"\n[ERROR] Unsupported LLM_PROVIDER '{provider}'. Supported values: openai, anthropic, gemini")
        sys.exit(1)

    req_body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=req_body, headers=headers, method="POST")

    max_retries = 3
    retry_delay = 2  # initial delay in seconds

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                
                try:
                    if provider == "openai":
                        msg = res_json["choices"][0]["message"]
                        content = msg.get("content") or ""
                        if not content and "reasoning_content" in msg:
                            content = msg["reasoning_content"]
                        if not content:
                            print("\n[DEBUG] Empty content in OpenAI response. Full response:")
                            print(json.dumps(res_json, indent=2))
                    elif provider == "anthropic":
                        content = res_json["content"][0]["text"]
                    elif provider == "gemini":
                        content = res_json["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as parse_err:
                    print("\n[ERROR] Failed to parse LLM response JSON:")
                    print(json.dumps(res_json, indent=2))
                    raise parse_err
                    
                return {
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            
            # Retry on transient server errors (500, 502, 503, 504) or rate limit (429)
            if e.code in [500, 502, 503, 504, 429] and attempt < max_retries - 1:
                import time
                print(f"\n[WARNING] LLM API returned status {e.code}. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            
            print(f"\n[ERROR] LLM API Request Failed ({provider.upper()}):")
            print(f"Status Code: {e.code}")
            print(f"Response: {err_body}")
            raise e
        except Exception as e:
            print(f"\n[ERROR] Unexpected error calling LLM API ({provider.upper()}): {e}")
            raise e
