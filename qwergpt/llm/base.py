from abc import ABC
from typing import List

from qwergpt.schema import Message
from qwergpt.llm.token_counter import TokenCounter


class LLM(ABC):
    token_counter = TokenCounter()

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def complete(self, messages: List[Message], max_tokens: int, metadata: bool) -> Message:
        pass

    def update_token_count(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.token_counter.update(self.name, prompt_tokens, completion_tokens, total_tokens)

    def get_token_stats(self):
        return self.token_counter.get_stats(self.name)

    @classmethod
    def get_total_token_stats(cls):
        return cls.token_counter.get_total_stats()
