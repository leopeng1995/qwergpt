from collections import defaultdict


class TokenCounter:
    def __init__(self):
        self.counters = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.last_call = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def update(self, llm_name: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        counter = self.counters[llm_name]
        counter["prompt_tokens"] += prompt_tokens
        counter["completion_tokens"] += completion_tokens
        counter["total_tokens"] += total_tokens
        
        # 更新最后一次调用的 tokens 统计
        self.last_call[llm_name] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    def get_stats(self, llm_name: str = None):
        if llm_name:
            return self.last_call[llm_name]
        return dict(self.last_call)

    def get_total_stats(self):
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for counter in self.counters.values():
            total["prompt_tokens"] += counter["prompt_tokens"]
            total["completion_tokens"] += counter["completion_tokens"]
            total["total_tokens"] += counter["total_tokens"]
        return total
