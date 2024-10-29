import io
import asyncio
import traceback
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr

from qwergpt.logs import logger


class RunCodeException(Exception):
    """自定义异常，用于代码执行错误"""
    pass


class BaseCoder(ABC):
    def __init__(self, question_id: int):
        self.global_context = {}
        self.question_id = question_id
        self.code_file_path = f'notebooks/{question_id}.py'
        self.executed_code = []
        self.latest_output = ''
        self.lock = asyncio.Lock()
    
    def get_executed_code(self):
        if len(self.executed_code) > 0:
            return self.executed_code[0]
        return ''
    
    @abstractmethod
    async def run_code(self, code: str, preserve_context=True):
        pass

    async def _execute_code(self, code, preserve_context):
        run_result = {
            'success': True,
            'ename': '',
            'evalue': '',
            'traceback': '',
            'result': '',
            'error': ''
        }

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                local_context = self.global_context.copy()
                try:
                    compiled_code = compile(code, '<string>', 'exec')
                    eval(compiled_code, local_context, local_context)
                    if 'main' in local_context and callable(local_context['main']):
                        local_context['main']()
                except Exception as e:
                    error_message = f"{type(e).__name__}: {str(e)}"
                    tb = traceback.format_exc()
                    raise RunCodeException(error_message, tb)

            if preserve_context:
                self.global_context.update(local_context)

        except RunCodeException as e:
            run_result.update({
                'success': False,
                'ename': 'RunCodeException',
                'evalue': str(e),
                'traceback': self._format_traceback(e.args[1], code)
            })
            logger.debug(f'Question {self.question_id} RunCodeException: {str(e)}')
        finally:
            run_result['result'] = stdout.getvalue()
            run_result['error'] = stderr.getvalue()

            stdout.close()
            stderr.close()

        if not preserve_context:
            self.global_context.clear()

        return run_result

    @staticmethod
    def _format_traceback(tb: str, code: str) -> str:
        tb_lines = tb.split('\n')
        code_lines = code.split('\n')
        for i, line in enumerate(tb_lines):
            if 'File "<string>"' in line:
                line_number = int(line.split(', line ')[1].split(',')[0])
                if 0 <= line_number - 1 < len(code_lines):
                    tb_lines[i] = f'  File "<string>", line {line_number}, in <module>'
                    tb_lines.insert(i + 1, f'    {code_lines[line_number - 1].strip()}')
        return '\n'.join(tb_lines)

    def clear_context(self) -> None:
        """清除保存的全局上下文"""
        self.global_context.clear()

    def get_latest_output(self) -> str:
        """获取最近一次代码执行的结果"""
        return self.latest_output

    def shutdown(self) -> None:
        """将执行过的代码写入文件"""
        try:
            with open(self.code_file_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(self.executed_code))
        except IOError as e:
            logger.debug(f"Error writing code to file: {e}")
