from abc import ABC, abstractmethod
from typing import List

from tenacity import (
    retry, 
    stop_after_attempt, 
    retry_if_exception
)

from qwergpt.llms import (
    ZhipuLLM,
    DeepSeekLLM,
)
from qwergpt.schema import (
    Task,
    Message,
)
from qwergpt.logs import logger
from qwergpt.graph import TaskGraph
from qwergpt.utils import (
    parse_code,
    should_retry,
)
from qwergpt.pipelines import PipelineComponent


class Plan(ABC):
    _question: str = ''
    _tasks: list[Task] = []
    _current_task_idx: int
    _current_task: Task = None
    _finished_tasks: list[Task] = []

    def __init__(self, question, tasks):
        self._question = question

        task_graph = TaskGraph()
        task_graph.add_tasks(tasks)

        # 获得拓扑排序后的任务列表
        self._tasks = task_graph.get_tasks()
        self._current_task_idx = 0
        self._current_task = self._tasks[self._current_task_idx]
    
    def finish_task(self, task: Task):
        self._current_task_idx += 1
        self.finished_tasks.append(task)

    @property
    def question(self):
        return self._question

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    @property
    def current_task(self) -> Task:
        return self._tasks[self._current_task_idx]
    
    @property
    def finished_tasks(self) -> list[Task]:
        return self._finished_tasks


class BasePlanner(PipelineComponent):
    def __init__(self, model: str = 'glm-4-air'):
        self._llm: ZhipuLLM = ZhipuLLM(model=model)

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception(should_retry))
    async def _get_instruction(self, messages: List[Message]) -> str:
        message = await self._llm.acomplete(messages)
        return parse_code(message.content, lang='json')
