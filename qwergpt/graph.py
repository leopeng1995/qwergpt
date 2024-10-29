import platform
from abc import ABC
from typing import List, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt

from qwergpt.schema import Task


class TaskGraph(ABC):
    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self.tasks: List[Task] = []

    def _topological_sort(self) -> None:
        topological_order = list(nx.topological_sort(self.G))
        self.tasks = [
            Task(
                task_id=node,
                instruction=self.G.nodes[node].get('instruction'),
                dependent_task_ids=self.G.nodes[node].get('dependent_task_ids'),
            )
            for node in topological_order
        ]

    def add_tasks(self, tasks: List[Dict[str, Any]]):
        for task in tasks:
            # 添加节点
            self.G.add_node(
                task['task_id'],
                label=f"Task {task['task_id']}",
                instruction=task['instruction'],
                dependent_task_ids=task['dependent_task_ids']
            )
            # 添加边
            for dep_id in task['dependent_task_ids']:
                self.G.add_edge(dep_id, task['task_id'])
        
        # 拓扑排序，确定任务执行顺序
        self._topological_sort()

    def get_tasks(self) -> list[Task]:
        return self.tasks
    
    @staticmethod
    def _set_font():
        system = platform.system()
        if system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        elif system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # 确保可以显示负号
        plt.rcParams['axes.unicode_minus'] = False

    def draw(self):
        # 设置中文显示字体
        self._set_font()

        # 绘制图
        pos = nx.spring_layout(self.G, k=1, iterations=50)  # 增加k值和迭代次数以扩大节点间距离
        plt.figure(figsize=(15, 10))  # 增加图的大小

        nx.draw(self.G, pos, node_color='lightblue', node_size=5000, arrows=True)

        # 绘制节点标签
        node_labels = nx.get_node_attributes(self.G, 'label')
        nx.draw_networkx_labels(self.G, pos, node_labels, font_size=8)

        # 显示图
        plt.title("Task Dependency DAG")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
