import asyncio
from together import Together
from .base import BaseLLMProvider, LLMResponse
from typing import List, Dict
from rich.console import Console

console = Console()

class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        self.client = Together()

    async def generate(self, messages: List[Dict], temperature=0.7, max_tokens=8192, task_type: str = "normal", **kwargs):
        try:
            kwargs = kwargs.copy()
            kwargs.pop("task_type", None)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            def call_together():
                return self.client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )

            response = await asyncio.to_thread(call_together)
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.dict() if hasattr(response, 'usage') else None,
            )
        except Exception as e:
            console.print(f"[red]Together Error: {e}[/]")
            raise

    async def embed(self, text):
        try:
            def call_together_embedding():
                return self.client.embeddings.create(
                    model="togethercomputer/m2-bert-80M-8k-retrieval",
                    input=text,
                )

            response = await asyncio.to_thread(call_together_embedding)
            return response.data[0]["embedding"] if isinstance(text, str) else [item["embedding"] for item in response.data]
        except Exception as e:
            console.print(f"[red]Embedding Error: {e}[/]")
            raise