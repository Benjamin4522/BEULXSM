from .groq import GroqProvider
from .mistral import MistralProvider
from .gemini import GeminiProvider
from .base import BaseLLMProvider
from typing import Dict
from rich.console import Console

console = Console()

class LLMRouter:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {
            "groq": GroqProvider(),
            "mistral": MistralProvider(),
            "gemini": GeminiProvider(),
        }

    async def generate(self, messages: list, task_type: str = "normal", **kwargs):
        task_type = task_type.lower()

        # Routing berdasarkan kemampuan provider
        if task_type in ["coding", "offensive", "payload", "exploit", "decomposition", "deep_reasoning", "vulnerability", "planner", "verification"]:
            provider_name = "gemini"
        elif task_type in ["fast", "tool_call", "simple", "chat", "normal"]:
            provider_name = "groq"
        else:
            provider_name = "mistral"

        provider = self.providers[provider_name]
        console.print(f"[dim]→ Using {provider_name.upper()} for task: {task_type}[/]")

        try:
            return await provider.generate(messages, task_type=task_type, **kwargs)
        except Exception as e:
            console.print(f"[yellow]{provider_name.upper()} failed: {e}[/]")
            fallback_order = ["groq", "gemini", "mistral"]
            for fallback_name in fallback_order:
                if fallback_name == provider_name:
                    continue
                fallback_provider = self.providers[fallback_name]
                console.print(f"[dim]→ Fallback to {fallback_name.upper()}[/]")
                try:
                    return await fallback_provider.generate(messages, task_type=task_type, **kwargs)
                except Exception as fallback_error:
                    console.print(f"[red]{fallback_name.upper()} fallback failed: {fallback_error}[/]")
            raise