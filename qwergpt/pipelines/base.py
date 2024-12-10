import json 
import asyncio
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Set, Callable, List


@dataclass
class PipelineData:
    data: Dict[str, Any] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        
    def update(self, data: Dict[str, Any]) -> None:
        self.data.update(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def debug(self, separator: str = " = ") -> str:
        output = ["Pipeline Data:"]
        for k, v in self.data.items():
            output.append(f"  {k}{separator}{v}")
            
        return "\n".join(output)


class PipelineComponent(ABC):
    """Base interface that all pipeline components must implement"""
    @abstractmethod
    async def run(self, pipeline_data: PipelineData) -> PipelineData:
        pass


class PipelineStatus(Enum):
    INIT = "initialized"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class Pipeline(ABC):
    def __init__(self, pipeline_id: str = None):
        self.id = pipeline_id
        self.status = PipelineStatus.INIT
        self.components: List[dict] = []
        self.observers: Set[Callable] = set()
        self.ws_server = None
        self.pipeline_data = None
    
    def set_ws_server(self, ws_server):
        self.ws_server = ws_server
    
    def add_observer(self, callback: Callable):
        self.observers.add(callback)
    
    def remove_observer(self, callback: Callable):
        self.observers.discard(callback)
    
    def notify_observers(self):
        status_data = {
            'pipelineId': self.id,
            'status': self.status.value,
            'components': self.components,
            'pipelineData': self.pipeline_data.to_dict() if self.pipeline_data else {}
        }
        status_json = json.dumps(status_data, ensure_ascii=False)
        
        for callback in self.observers:
            callback(status_json)
        
        if self.ws_server and self.id:
            asyncio.create_task(self.ws_server.notify_pipeline_status(self.id, status_json))
    
    async def start(self, *args, **kwargs) -> Any:
        self.status = PipelineStatus.RUNNING
        self.notify_observers()
        try:
            result = await self.run(*args, **kwargs)
            self.status = PipelineStatus.COMPLETED
            self.notify_observers()
            return result
        except Exception as e:
            self.status = PipelineStatus.ERROR
            self.notify_observers()
            raise e
    
    async def pause(self):
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
    
    async def resume(self):
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
    
    def log_component_metrics(self, component_name: str, execution_time: float):
        component = {
            'name': component_name,
            'execution_time': execution_time
        }
        self.components.append(component)

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        pass
