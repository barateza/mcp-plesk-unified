import json
import logging
import os
import requests
from typing import Optional, List

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "stepfun/step-3-5-flash:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
]


class AIClient:
    """A thin wrapper for LLM calls with retry/fallback policy."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set. AI calls will fail.")

    def generate_description(
        self, text: str, model_list: Optional[List[str]] = None
    ) -> str:
        """
        Attempts to get a description using a tiered fallback system.
        """
        if not text.strip():
            return "File unreadable."

        models = model_list if model_list is not None else DEFAULT_MODELS

        for model in models:
            try:
                content = (
                    "Summarize the technical purpose of the following text "
                    "in exactly one concise sentence.\n\n" + text
                )

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 100,
                }

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=15,
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                else:
                    logger.debug(
                        f"Model {model} returned status {response.status_code}"
                    )
            except Exception as e:
                logger.debug(f"Model {model} failed with exception: {e}")
                continue

        return "Description unavailable."
