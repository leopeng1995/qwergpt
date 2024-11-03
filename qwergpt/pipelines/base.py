import json 
import asyncio
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Set, Callable, List


@dataclass
class PipelineData:
    """统一的Pipeline数据结构"""
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        
    def update(self, data: Dict[str, Any]) -> None:
        self.data.update(data)
        
    def get_meta(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)
    
    def set_meta(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def debug(self, separator: str = " = ") -> str:
        output = ["Pipeline Data:"]
        for k, v in self.data.items():
            output.append(f"  {k}{separator}{v}")
            
        output.append("\nPipeline Metadata:")
        for k, v in self.metadata.items():
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
        self._observers: Set[Callable] = set()
        self._ws_server = None
        self._component_order = 0
    
    def set_ws_server(self, ws_server):
        self._ws_server = ws_server
    
    def add_observer(self, callback: Callable):
        self._observers.add(callback)
    
    def remove_observer(self, callback: Callable):
        self._observers.discard(callback)
    
    def _notify_observers(self):
        status_data = {
            'pipeline_id': self.id,
            'status': self.status.value,
            'components': self.components
        }
        print(status_data)
        status_json = json.dumps(status_data)
        
        for callback in self._observers:
            callback(status_json)
        
        if self._ws_server and self.id:
            asyncio.create_task(self._ws_server.notify_pipeline_status(self.id, status_json))
    
    async def start(self, *args, **kwargs) -> Any:
        self.status = PipelineStatus.RUNNING
        self._notify_observers()
        try:
            result = await self.run(*args, **kwargs)
            self.status = PipelineStatus.COMPLETED
            self._notify_observers()
            return result
        except Exception as e:
            self.status = PipelineStatus.ERROR
            self._notify_observers()
            raise e
    
    async def pause(self):
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
    
    async def resume(self):
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
    
    def log_component_metrics(self, component_name: str, input_data: Any, output_data: Any, execution_time: float):
        component = {
            'name': component_name,
            'order': self._component_order
        }
        self.components.append(component)
        self._component_order += 1

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        pass
