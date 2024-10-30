from abc import ABC


class Prompt(ABC):
    def __init__(self, template: str):
        self.template = template
    
    def compose(self, **kwargs) -> str:
        return self.template.format(**kwargs)
