from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class BaseLLMProvider(ABC):
    """Base class untuk semua LLM Provider di BEULXSM"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      response_format: Optional[BaseModel] = None, 
                      tools: Optional[List[Dict]] = None, temperature: float = 0.7):
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Optional[Dict] = None

class BaseLLMProvider(ABC):
    """Base Provider untuk semua LLM di BEULXSM"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[BaseModel] = None,
        tools: Optional[List[Dict]] = None,
        task_type: str = "normal"
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def embed(self, text: str | List[str]) -> List[List[float]]:
        pass
    from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Optional[Dict] = None

class BaseLLMProvider(ABC):
    """Base Provider untuk semua LLM di BEULXSM"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[BaseModel] = None,
        tools: Optional[List[Dict]] = None,
        task_type: str = "normal"
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def embed(self, text: str | List[str]) -> List[List[float]]:
        pass