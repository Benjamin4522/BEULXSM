import litellm
from .base import BaseLLMProvider, LLMResponse
from typing import List, Dict, Optional
from rich.console import Console

console = Console()

class GroqProvider(BaseLLMProvider):
    """Groq Provider - Super fast untuk planning & execution"""

    def __init__(self):
        litellm.verbose = False

    async def generate(self, messages: List[Dict], temperature=0.7, max_tokens=4096, task_type: str = "normal", **kwargs):
        try:
            kwargs = kwargs.copy()
            kwargs.pop("task_type", None)
            response = await litellm.acompletion(
                model="groq/llama-3.3-70b-versatile",   # Bisa diganti mistral model kalau ada di Groq
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            usage_value = getattr(response, 'usage', None)
            if hasattr(usage_value, 'dict'):
                usage_value = usage_value.dict()

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=usage_value
            )
        except Exception as e:
            console.print(f"[red]Groq Error: {e}[/]")
            raise

    async def embed(self, text: str | List[str]):
        """Groq tidak support embedding → fallback ke Gemini"""
        console.print("[yellow]Groq tidak support embedding, fallback ke Gemini...[/]")
        from .gemini import GeminiProvider
        gemini = GeminiProvider()
        return await gemini.embed(text)