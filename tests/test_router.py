import asyncio
import llm.providers.router as router_module


class DummyProvider:
    def __init__(self, name):
        self.name = name

    async def generate(self, messages, **kwargs):
        return {"provider": self.name, "messages": messages, **kwargs}


def make_router():
    router_module.GroqProvider = lambda: DummyProvider("groq")
    router_module.MistralProvider = lambda: DummyProvider("mistral")
    router_module.GeminiProvider = lambda: DummyProvider("gemini")
    router = router_module.LLMRouter()
    return router


def test_router_uses_groq_for_fast_tasks():
    router = make_router()
    result = asyncio.run(router.generate([{"role": "system", "content": "Hello"}], task_type="fast"))
    assert result["provider"] == "groq"


def test_router_uses_gemini_for_coding_tasks():
    router = make_router()
    result = asyncio.run(router.generate([{"role": "system", "content": "Hello"}], task_type="coding"))
    assert result["provider"] == "gemini"


def test_router_uses_mistral_for_other_tasks():
    router = make_router()
    result = asyncio.run(router.generate([{"role": "system", "content": "Hello"}], task_type="balanced"))
    assert result["provider"] == "mistral"
