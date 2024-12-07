import json
from typing import (
    List, 
    Optional,
    AsyncGenerator
)

import aiohttp
import asyncio
import requests

from qwergpt.schema import Message
from qwergpt.llms.base import LLM
from qwergpt.llms.errors import LLMAPIUnknownError


class OllamaLLM(LLM):
    _semaphore = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_semaphore(cls):
        async with cls._lock:
            if cls._semaphore is None:
                cls._semaphore = asyncio.Semaphore(20)
        return cls._semaphore

    def __init__(self, model='llama3', base_url='http://localhost:11434/api/generate') -> None:
        super().__init__("OllamaLLM")
        self.model = model
        self.base_url = base_url

    def _get_headers(self):
        return {
            'Content-Type': 'application/json',
        }

    def _prepare_request_data(self, messages: List[Message], max_tokens: int, stream: bool = False, context_window: int = 2048) -> dict:
        prompt = messages[-1].content
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "max_tokens": max_tokens,
            "options": {
                "num_ctx": context_window,
            }
        }
    
    def complete(self, messages: List[Message], max_tokens=1024, metadata=False) -> Message:        
        headers = self._get_headers()
        data = self._prepare_request_data(messages, max_tokens)

        response = requests.post(self.base_url, headers=headers, json=data)
        res = response.json()
        if 'error' in res:
            raise LLMAPIUnknownError(res['error'])

        content = res['response']
        return Message(role='assistant', content=content)

    async def acomplete(self, messages: List[Message], max_tokens: int = 4095, metadata=False) -> Message:
        semaphore = await self.get_semaphore()
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                data = self._prepare_request_data(messages, max_tokens)

                timeout = aiohttp.ClientTimeout()
                async with session.post(self.base_url, headers=headers, json=data, timeout=timeout) as response:
                    res = await response.json()

                if 'error' in res:
                    raise LLMAPIUnknownError(res['error'])

                content = res['response']
                return Message(role='assistant', content=content)

    async def acomplete_stream(self, messages: List[Message], max_tokens: int = 4095, context_window: int = 2048, metadata=False) -> AsyncGenerator[Message, None]:
        semaphore = await self.get_semaphore()
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                data = self._prepare_request_data(messages, max_tokens, stream=True, context_window=context_window)
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    async for line in response.content:
                        message = self._process_stream_line(line)
                        if message:
                            yield message

    def _process_stream_line(self, line: bytes) -> Optional[Message]:
        try:
            chunk = line.decode('utf-8').strip()
            if chunk:
                chunk_data = json.loads(chunk)
                content = chunk_data['response']
                if content:
                    return Message(role='assistant', content=content)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {chunk}")
        except Exception as e:
            print(f"Error processing stream: {str(e)}")
        return None
