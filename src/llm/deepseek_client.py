# src/llm/deepseek_client.py

# src/llm/deepseek_client.py

from __future__ import annotations

import os
from typing import Dict, Any

import requests
from dotenv import load_dotenv  # <-- tambahkan ini

# load file .env di root project
load_dotenv()  # <-- dan ini


class DeepSeekClient:
    """
    Klien sederhana untuk memanggil DeepSeek (mis. deepseek-chat / V3) via API HTTP.
    Config diambil dari environment (.env):
      - DEEPSEEK_API_KEY
      - DEEPSEEK_API_URL   (default: https://api.deepseek.com/chat/completions)
      - DEEPSEEK_MODEL     (default: deepseek-chat)
    """

    def __init__(
        self,
        model_env: str = "DEEPSEEK_MODEL",
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_url_env: str = "DEEPSEEK_API_URL",
    ) -> None:
        self.api_key = os.getenv(api_key_env)
        self.api_url = os.getenv(api_url_env) or "https://api.deepseek.com/chat/completions"
        self.model_name = os.getenv(model_env) or "deepseek-chat"

        if not self.api_key:
            raise RuntimeError(
                f"DeepSeekClient: env {api_key_env} belum diset. "
                "Silakan isi di file .env."
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 48,
        temperature: float = 0.2,
    ) -> str:
        """
        Panggil endpoint /chat/completions DeepSeek dan kembalikan teks jawaban.
        Format ini mengikuti dokumentasi resmi DeepSeek (OpenAI-compatible). 
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-style response
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as e:  # buat debug kalau error format
            raise RuntimeError(f"Gagal parsing respons DeepSeek: {e}\nResponse: {data}")

        return text.strip()
