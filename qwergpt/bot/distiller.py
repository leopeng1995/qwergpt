from abc import ABC

from tenacity import retry, stop_after_attempt, retry_if_exception

from qwergpt.llms import ZhipuLLM
from qwergpt.schema import Message
from qwergpt.utils import should_retry


DISTILLER_SYSTEM_PROMPT: str = """你是信息蒸馏领域高度专业和智能的专家，你擅长从用户输入查询中提取关键信息以解决问题。"""

DISTILLER_PROMPT_TEMPLATE: str = """
作为信息蒸馏领域高度专业和智能的专家，你擅长从用户输入查询中提取关键信息以解决问题。你能够熟练地将提取的信息转化为适合相应问题类型的格式。如果问题可以泛化到更高层次以解决多个问题，你将在下一个回应中提供进一步的分析和解释。

[数据表定义]
{database_schema}

[工具列表]
{tools_desc}

[任务]
请从用户输入的查询中分类并提取解决问题所需的关键信息。将这两个元素结合起来，生成提炼后的信息。随后，根据问题类型，将这些提炼的信息传递给你的下游元规划器。问题类型应属于上述六个类别之一，提炼的信息应包括：

1. 从用户输入中提取的关键变量的值和信息，这些信息将交给相应的专家进行任务解决，确保提供解决问题所需的所有重要信息。
2. 问题的目标和相应的约束条件。
3. 基于1和2扩展问题，提出一个可以解决用户查询并处理更多输入和输出变化的元问题。结合扩展问题的现实场景以及原始问题中关键变量的类型和信息约束，限制扩展问题中的关键变量。之后，使用用户查询输入的关键信息作为输入，以解决问题为例。
4. 尝试将问题转化为Python算法问题，并提供输入参数。
5. 你的任务是提炼问题，不应在回答中给出最终结果或可能的解决方案。

请按照以下格式提炼信息，并在输出提炼信息后停止回应。

1. 关键信息：
2. 限制：（应当注意，答案应严格遵循现实世界规则，如算术方程中的运算符优先级、括号的需要等。因此，根据提炼的信息，强调问题中需要遵循的现实世界规则。）
3. 提炼后的任务：

**生成到此结束。请勿在答案中显示此消息**

用户输入:
{question}
"""


class Distiller(ABC):

    _llm: ZhipuLLM

    def __init__(self):
        self._llm = ZhipuLLM()

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception(should_retry))
    async def run(self, question, database_schema, tools_desc):
        distill_prompt = DISTILLER_PROMPT_TEMPLATE.format(
            question=question,
            database_schema=database_schema,
            tools_desc=tools_desc
        )
        messages = [
            Message(role='system', content=DISTILLER_SYSTEM_PROMPT),
            Message(role='user', content=distill_prompt),
        ]
        message = await self._llm.acomplete(messages)

        distilled_information = message.content
        return distilled_information
