from __future__ import annotations

import os
from dataclasses import dataclass

import dspy
from openai import OpenAI


@dataclass
class OpenAICompatConfig:
    model: str
    api_base: str
    api_key: str
    label: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAICompatConfig | None":
        model = os.getenv("CHATDKU_LLM_MODEL")
        api_base = os.getenv("CHATDKU_LLM_BASE_URL")
        api_key = os.getenv("CHATDKU_LLM_API_KEY", "local")
        if not model or not api_base:
            return None
        return cls(model=model, api_base=api_base, api_key=api_key)

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "OpenAICompatConfig":
        return cls(
            model=payload["model"],
            api_base=payload["api_base"],
            api_key=payload.get("api_key", "local"),
            label=payload.get("label"),
        )

    def display_name(self) -> str:
        return self.label or self.model


class OpenAICompatClient:
    def __init__(self, config: OpenAICompatConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )

    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def healthcheck(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


def build_dspy_lm(config: OpenAICompatConfig | None) -> dspy.LM | None:
    if config is None:
        return None
    return dspy.LM(
        f"openai/{config.model}",
        api_base=config.api_base,
        api_key=config.api_key,
        model_type="chat",
    )
