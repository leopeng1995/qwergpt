from abc import ABC, abstractmethod

from qwergpt.llm import (
    ZhipuLLM,
    DeepSeekLLM,
)


class BaseRewriter(ABC):
    def __init__(self):
        self._llm = ZhipuLLM()

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_user_prompt_template(self) -> str:
        pass
