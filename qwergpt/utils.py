import re
import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from pdf2image import convert_from_path

from qwergpt.logs import logger
from qwergpt.llm.errors import LLMBalanceDepletionError
from asyncio.exceptions import CancelledError, TimeoutError


def should_retry(exception):
    # 以下异常类型需要跳过重试
    return not isinstance(exception, (LLMBalanceDepletionError, TimeoutError, CancelledError))


def parse_code(text: str, lang: str = ""):
    pattern = rf"```{lang}.*?\s+(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        return text  # just assume original text is code
    return code


def parse_python(text: str, lang: str = ""):
    pattern = rf"```{lang}.*?\s+(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1)
        return code
    
    text = text.strip()
    lines = text.split('\n')
    if lines[0].startswith('python'):
        if not text.startswith('```'):
            text = '```' + text
        if not text.endswith('```'):
            text += '\n```'
    elif lines[0].startswith('```python'):
        if not text.endswith('```'):
            text += '\n```'
    elif lines[0].startswith('# 你修改后的 Python 代码') \
        or lines[0].startswith('# 修改后的 Python 代码'):
        text = f'```python\n{text}'
        if not text.endswith('```'):
            text += '\n```'

    pattern = rf"```{lang}.*?\s+(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1)
        return code
    
    if text.strip().startswith('from app.law.tools import *'):
        code = text
        return code

    logger.debug(f"parse_python not match following text: {text}")
    raise ValueError('parse_python')


def parse_json(text: str):
    def try_parse(json_text):
        try:
            # 首先尝试直接解析
            return json.loads(json_text)
        except json.JSONDecodeError:
            try:
                # 去掉单行注释
                json_text = re.sub(r'//.*', '', json_text)
                # 去掉多行注释
                json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
                # 去掉行尾注释（包括 # 开头的注释）
                json_text = re.sub(r'#.*', '', json_text)
                # 去掉空行
                json_text = '\n'.join(line for line in json_text.splitlines() if line.strip())
                return json.loads(json_text)
            except json.JSONDecodeError:
                return None

    # 尝试直接解析
    data = try_parse(text)
    if data:
        return data

    # 尝试使用 parse_code 函数
    parsed_text = parse_code(text, lang='json')
    data = try_parse(parsed_text)
    if data:
        return data

    # 尝试提取 JSON 部分
    start, end = text.find('['), text.rfind(']')
    if start != -1 and end != -1 and start < end:
        json_part = text[start:end+1]
        data = try_parse(json_part)
        if data:
            return data

    # 尝试处理特殊情况
    if text.strip().startswith("},"):
        new_text = "[" + text.strip()[2:]
        data = try_parse(new_text)
        if data:
            return data
        
        new_text = new_text + "]"
        data = try_parse(new_text)
        if data:
            return data
        
    # 尝试处理最后两个右括号可能分布在不同行的情况
    lines = text.strip().split('\n')
    if len(lines) >= 2 and lines[-1].strip() == ']' and lines[-2].strip().endswith(']'):
        new_text = '\n'.join(lines[:-1])
        data = try_parse(new_text)
        if data:
            return data
        
    # 使用正则表达式提取 JSON 部分
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        new_text = match.group()
        data = try_parse(new_text)
        if data:
            return data

    parsed_text = parse_code(text, lang='json')
    new_text = f'[{parsed_text}]'
    data = try_parse(new_text)
    if data:
        return data
    
    if text.strip().startswith("},"):
        new_text = "[" + text.strip()[2:]
        if new_text.endswith('```'):
            new_text = f"```json\n{new_text}"
            parsed_text = parse_code(new_text, lang='json')
            data = try_parse(parsed_text)
            if data:
                return data

    # 所有尝试都失败
    logger.debug(f"parse_json not match following text: {text}")
    raise ValueError("parse_json failed")


def parse_markdown(text: str):
    def try_parse(text):
        pattern = rf"```markdown.*?\s+(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
            return code

        return None

    data = try_parse(text)
    if data:
        return data
    
    text = text.strip()
    lines = text.split('\n')
    if lines[0].startswith('```markdown'):
        if not text.endswith('```'):
            text += '\n```'
    else:
        text = '````markdown\n' + text
        if not text.endswith('```'):
            text += '\n```'
    
    data = try_parse(text)
    if data:
        return data

    logger.debug(f"parse_markdown not match following text: {text}")
    raise ValueError('parse_markdown')


def format_filtered_tables(filtered_tables):
    formatted_output = ""
    for table in filtered_tables:
        fields = table['fields']
        if isinstance(fields, str):
            fields_str = fields
        elif isinstance(fields, list):
            fields_str = ', '.join(fields)
        else:
            fields_str = str(fields)
        
        formatted_output += f"表名: {table['table_name']}\n字段: {fields_str}\n\n"
        formatted_output += "-------------------------------------\n\n"
    return formatted_output.strip()


# 字符串: 是否包含至少一个数字
def has_digits(string):
    return any(char.isdigit() for char in string)


# 字符串: 没有数字
def no_digits(string):
    return not has_digits(string)


# 字符串: 全是数字
def all_digits(string):
    return string.isdigit()


# 字符串: 不包含中文
def no_chinese(string):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return not bool(pattern.search(string))


def convert_date_format(date_string):
    try:
        date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    except:
        date_object = datetime.strptime(date_string, "%Y-%m-%d")
    
    year = date_object.year
    month = date_object.month
    day = date_object.day
    
    # 构建中文日期字符串，确保月份没有前导零
    chinese_date = f"{year}年{month}月{day}日"
    return chinese_date


# 合并 API Summary
def group_api_calls(summary):
    # 创建一个字典来存储每个任务的完整信息
    tasks = {task['task_id']: task for task in summary}
    
    # 创建一个函数来获取任务及其所有依赖
    def get_task_with_dependencies(task_id, visited=None):
        if visited is None:
            visited = set()
        if task_id in visited:
            return []
        visited.add(task_id)
        task = tasks[task_id]
        result = [task]
        for dep_id in task['dependent_task_ids']:
            result = get_task_with_dependencies(dep_id, visited) + result
        return result

    # 创建分组结果
    grouped_calls = []
    for task in reversed(summary):  # 从后向前遍历，以便先处理最深的依赖
        if not any(task in group for group in grouped_calls):
            group = get_task_with_dependencies(task['task_id'])
            # 检查是否存在包含当前组的超集
            superset_exists = False
            for existing_group in grouped_calls:
                if set(t['task_id'] for t in group).issubset(set(t['task_id'] for t in existing_group)):
                    superset_exists = True
                    break
            if not superset_exists:
                grouped_calls.append(group)
    
    return grouped_calls


# 相比 group_api_calls，过滤了 used_tool 为空的情况
def group_api_calls2(summary):
    # 创建一个字典来存储每个有效任务的完整信息
    tasks = {task['task_id']: task for task in summary if task['used_tool']}
    
    # 创建一个函数来获取任务及其所有依赖
    def get_task_with_dependencies(task_id, visited=None):
        if visited is None:
            visited = set()
        if task_id not in tasks or task_id in visited:
            return []
        visited.add(task_id)
        task = tasks[task_id]
        result = [task]
        for dep_id in task['dependent_task_ids']:
            result = get_task_with_dependencies(dep_id, visited) + result
        return result

    # 创建分组结果
    grouped_calls = []
    for task in reversed(summary):  # 从后向前遍历，以便先处理最深的依赖
        if task['used_tool'] and not any(task in group for group in grouped_calls):
            group = get_task_with_dependencies(task['task_id'])
            # 检查是否存在包含当前组的超集
            superset_exists = False
            for existing_group in grouped_calls:
                if set(t['task_id'] for t in group).issubset(set(t['task_id'] for t in existing_group)):
                    superset_exists = True
                    break
            if not superset_exists:
                grouped_calls.append(group)
    
    return grouped_calls


# 统计串行调用次数
def count_serial_api_calls(summary):
    total_serial_calls = 0
    
    for group in summary:
        serial_calls = sum(1 for task in group if task['dependent_task_ids'])
        total_serial_calls += serial_calls
    
    return total_serial_calls


# 统计串行调用了多少个API
def count_unique_serial_api_calls(summary):
    serial_api_calls = set()
    
    for group in summary:
        for task in group:
            if task['dependent_task_ids']:  # 如果dependent_task_ids不为空
                serial_api_calls.add(task['used_tool'])
    
    # 第一个漏掉了
    return len(serial_api_calls) + 1


# 总共调用了多少类
def count_unique_api_tools(summary):
    unique_tools = set()
    
    for chain in summary:
        for task in chain:
            if task['used_tool'] != '' and task['used_tool'] != 'None':
                unique_tools.add(task['used_tool'])
    
    return len(unique_tools)


# 统计总调用次数
# 注意这个 summary 是原始的 summary
def count_api_calls(summary):
    total_calls = 0

    for task in summary:
        if task['used_tool'] != '' and task['used_tool'] != 'None':
            total_calls += 1
    
    return total_calls


def convert_single_pdf(pdf_path: Path, output_root: Path) -> None:
    """转换单个PDF文件为图片"""
    try:
        # 创建输出目录
        output_folder = output_root / pdf_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 转换PDF为图像
        images = convert_from_path(str(pdf_path))
        
        # 创建一个内部进度条
        pbar = tqdm(
            total=len(images),
            desc=f"保存 {pdf_path.name} 的图片",
            leave=False  # 处理完当前PDF后清除这个进度条
        )
        
        def save_image(args):
            idx, image = args
            image.save(str(output_folder / f'page_{idx}.png'), 'PNG')
            pbar.update(1)
        
        # 使用线程池并行保存图像
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(save_image, enumerate(images, 1))
            
        pbar.close()
        logging.info(f"成功转换 {pdf_path.name}")
        
    except Exception as e:
        logging.error(f"转换 {pdf_path.name} 时出错: {str(e)}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_input_file(file_path: str, last_position: int = 0):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()[last_position:]


def append_to_json(filename, item):
    with open(filename, 'a', encoding='utf-8') as file:
        json_string = json.dumps(item, ensure_ascii=False)
        file.write(json_string + '\n')
