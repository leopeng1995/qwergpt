from typing import Annotated

from pydantic import BaseModel


# 思维模板
class ThoughtTemplate(BaseModel):

    desc: Annotated[str, "模板描述"]
    template: Annotated[str, "模板内容"]
    category: Annotated[str, "模板类别"]
