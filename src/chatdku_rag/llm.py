from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LocalLLM:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "LocalLLM | None":
        base_url = os.getenv("CHATDKU_LLM_BASE_URL")
        model = os.getenv("CHATDKU_LLM_MODEL")
        api_key = os.getenv("CHATDKU_LLM_API_KEY", "EMPTY")
        if not base_url or not model:
            return None
        return cls(base_url=base_url, model=model, api_key=api_key)

    def complete(self, prompt: str) -> str | None:
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None

        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError):
            return None
