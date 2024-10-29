from abc import ABC


class SolutionSpace(ABC):

    _question_id: str
    _result: str
    _executed_code: str

    def __init__(self, question_id):
        self._question_id = question_id
        self._result = ''

    def set_result(self, result):
        self._result = result

    def get_result(self):
        return self._result
    
    def set_executed_code(self, executed_code):
        self._executed_code = executed_code

    def get_executed_code(self):
        return self._executed_code
