import litellm
from .base import BaseLLMProvider, LLMResponse
from typing import List, Dict

class MistralProvider(BaseLLMProvider):
    async def generate(self, messages: List[Dict], temperature=0.7, max_tokens=4096, task_type: str = "normal", **kwargs):
        kwargs = kwargs.copy()
        kwargs.pop("task_type", None)
        response = await litellm.acompletion(
            model="mistral/mistral-large-latest",   # atau mistral/mistral-small-latest
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.dict() if hasattr(response, 'usage') else None
        )

    async def embed(self, text):
        # Mistral support embedding
        response = await litellm.aembedding(
            model="mistral/mistral-embed",
            input=text
        )
        return response.data[0]["embedding"] if isinstance(text, str) else [d["embedding"] for d in response.data]