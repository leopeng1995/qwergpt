from dataclasses import dataclass
from typing import Dict, List, Any, Callable


@dataclass
class EvaluationStep:
    name: str
    metric_fn: Callable
    weight: float = 1.0


class EvaluationConfig:
    def __init__(self):
        self.steps: Dict[str, EvaluationStep] = {}
        
    def add_step(self, step: EvaluationStep):
        self.steps[step.name] = step
        
    def get_step(self, name: str) -> EvaluationStep:
        return self.steps.get(name)



class PipelineEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def evaluate_step(self, step_name: str, predicted: Any, ground_truth: Any) -> Dict[str, float]:
        step = self.config.get_step(step_name)
        if not step:
            return {}
            
        metrics = step.metric_fn(predicted, ground_truth)
        if metrics is None:
            return {}
            
        return {f"{step_name}_{k}": v * step.weight for k, v in metrics.items()}
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not all_metrics:
            return {}
            
        # 聚合所有指标
        aggregated = {}
        for metrics in all_metrics:
            if not metrics:  # 跳过空字典
                continue
            for k, v in metrics.items():
                if k not in aggregated:
                    aggregated[k] = []
                aggregated[k].append(v)
                
        # 如果没有有效指标，返回空字典
        if not aggregated:
            return {}
            
        return {k: sum(v)/len(v) for k, v in aggregated.items()}
