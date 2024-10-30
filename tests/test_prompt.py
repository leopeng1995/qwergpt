from qwergpt.prompt import Prompt


PROMPT_TEMPLATE: str = """{initial} {hijack} {target}"""

QUERY_PROMPT = Prompt(PROMPT_TEMPLATE)

query = QUERY_PROMPT.compose(
    initial="初始提示",
    hijack="劫持提示",
    target="目标提示"
)

print(query)
