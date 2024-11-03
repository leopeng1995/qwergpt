import json
import asyncio
from typing import Dict, Set

import websockets


class PipelineWebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.pipeline_states: Dict[str, str] = {}
    
    async def register(self, websocket: websockets.WebSocketServerProtocol, pipeline_id: str):
        if pipeline_id not in self.connections:
            self.connections[pipeline_id] = set()
        self.connections[pipeline_id].add(websocket)
        
        # 如果存在该pipeline的状态，立即发送给新连接的客户端
        if pipeline_id in self.pipeline_states:
            try:
                await websocket.send(self.pipeline_states[pipeline_id])
            except websockets.ConnectionClosed:
                await self.unregister(websocket, pipeline_id)
    
    async def unregister(self, websocket: websockets.WebSocketServerProtocol, pipeline_id: str):
        if pipeline_id in self.connections:
            self.connections[pipeline_id].discard(websocket)
            if not self.connections[pipeline_id]:
                del self.connections[pipeline_id]
    
    async def notify_pipeline_status(self, pipeline_id: str, status_data: str):
        self.pipeline_states[pipeline_id] = status_data
        
        if pipeline_id in self.connections:
            websockets_to_remove = set()
            for websocket in self.connections[pipeline_id]:
                try:
                    await websocket.send(status_data)
                except websockets.ConnectionClosed:
                    websockets_to_remove.add(websocket)
            
            for websocket in websockets_to_remove:
                await self.unregister(websocket, pipeline_id)
    
    async def handler(self, websocket: websockets.WebSocketServerProtocol):
        try:
            message = await websocket.recv()
            data = json.loads(message)
            pipeline_id = data.get('pipeline_id')
            
            if not pipeline_id:
                await websocket.send(json.dumps({"error": "No pipeline_id provided"}))
                return
            
            await self.register(websocket, pipeline_id)
            
            try:
                await websocket.wait_closed()
            except websockets.ConnectionClosed:
                pass
                
        except websockets.ConnectionClosed:
            if 'pipeline_id' in locals():
                await self.unregister(websocket, pipeline_id)
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            if 'pipeline_id' in locals():
                await self.unregister(websocket, pipeline_id)
    
    async def start(self):
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()
