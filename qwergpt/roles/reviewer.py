from abc import ABC, abstractmethod

from tenacity import (
    retry, 
    stop_after_attempt, 
    retry_if_exception
)

from qwergpt.llms import (
    ZhipuLLM,
    DeepSeekLLM,
)
from qwergpt.schema import Message
from qwergpt.utils import (
    parse_code, 
    should_retry
)
from qwergpt.logs import logger


class BaseReviewer(ABC):

    def __init__(self, model: str = 'glm-4-air'):
        self._llm = ZhipuLLM(model=model)

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_user_prompt_template(self) -> str:
        pass

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception(should_retry))
    def run(self, question: str, answer: str):
        review_prompt = self.get_user_prompt_template().format(
            question=question,
            answer=answer
        )
        messages = [
            Message(role='system', content=self.get_system_prompt()),
            Message(role='user', content=review_prompt),
        ]
        message = self._llm.complete(messages)
        review_comment = parse_code(text=message.content, lang='json')

        return review_comment
