import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random


class RacetrackEnv:
    """
    赛道环境类
    状态: (x, y, vx, vy)
    动作: (ax, ay) 其中 ax, ay ∈ {-1, 0, 1}
    """
    
    def __init__(self, track_size=(32, 17), max_speed=5):
        self.track_size = track_size
        self.max_speed = max_speed
        self.min_speed = 0  # 恢复：速度严格非负
        
        # 创建赛道地图 (0:空地, 1:墙, 2:起点, 3:终点)
        self.track = self._create_track()
        self.start_positions = self._get_start_positions()
        self.goal_positions = self._get_goal_positions()
        
        # 动作空间: 9个动作 (-1,-1), (-1,0), ..., (1,1)
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.n_actions = len(self.actions)
        
        # 当前状态
        self.state = None
        self.reset()
    
    def _create_track(self) -> np.ndarray:
        """恢复用户原来的赛道布局"""
        track = np.ones(self.track_size, dtype=int)  # 默认全是墙
        
        # 第一列（索引0）：第5行到第14行是赛道（索引4到13）
        track[4:14, 0] = 0
        
        # 第二列（索引1）：第4行到第22行是赛道（索引3到21）
        track[3:22, 1] = 0
        
        # 第三列（索引2）：第2行到第29行是赛道（索引1到28）
        track[1:29, 2] = 0
        
        # 第四列到第九列（索引3到8）：全部都是赛道
        track[:, 3:9] = 0
        
        # 第十列（索引9）：第1行到第7行是赛道（索引0到6）
        track[0:7, 9] = 0
        
        # 第十一列到第十七列（索引10到16）：第1行到第6行是赛道（索引0到5）
        track[0:6, 10:17] = 0
        
        # 最后一行（索引31）全是起点
        track[31, :] = 2
        
        # 最后一列（索引16）的前6行全是终点（索引0到5）
        track[0:6, 16] = 3
        
        return track
    
    def _get_start_positions(self) -> List[Tuple[int, int]]:
        """获取所有起点位置"""
        positions = []
        for i in range(self.track_size[0]):
            for j in range(self.track_size[1]):
                if self.track[i, j] == 2:
                    positions.append((i, j))
        return positions
    
    def _get_goal_positions(self) -> List[Tuple[int, int]]:
        """获取所有终点位置"""
        positions = []
        for i in range(self.track_size[0]):
            for j in range(self.track_size[1]):
                if self.track[i, j] == 3:
                    positions.append((i, j))
        return positions
    
    def reset(self) -> Tuple[int, int, int, int]:
        """重置环境到起点"""
        start_pos = random.choice(self.start_positions)
        self.state = (start_pos[0], start_pos[1], 0, 0)  # (x, y, vx, vy)
        return self.state
    
    def step(self, action_idx: int) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """
        执行一步动作
        返回: (next_state, reward, done)
        """
        x, y, vx, vy = self.state
        ax, ay = self.actions[action_idx]
        
        # 题目要求：0.1概率速度保持不变
        if random.random() < 0.1:
            ax, ay = 0, 0
        
        # 更新速度（严格非负，重新定义方向意义）
        # vx > 0: 向上移动, vy > 0: 向右移动  
        new_vx = max(0, min(self.max_speed, vx + ax))
        new_vy = max(0, min(self.max_speed, vy + ay))
        
        # 题目要求：速度不能同时为零（除了在起点）
        if new_vx == 0 and new_vy == 0 and (x, y) not in self.start_positions:
            # 如果当前不在起点且速度将变为零，设置最小速度
            new_vx = 1  # 向上移动
            new_vy = 1  # 向右移动
        
        # 计算新位置 - 重新定义坐标移动方向
        # vx > 0 表示向上移动（x坐标减小）
        # vy > 0 表示向右移动（y坐标增大）
        new_x = x - new_vx  # 向上移动时x减小
        new_y = y + new_vy  # 向右移动时y增大
        
        # 检查碰撞
        if self._check_collision(x, y, new_x, new_y):
            # 碰撞：重置到起点
            self.state = self.reset()
            return self.state, -10, False  # 减少碰撞惩罚
        
        # 检查是否到达终点
        if (new_x, new_y) in self.goal_positions:
            self.state = (new_x, new_y, new_vx, new_vy)
            return self.state, 100, True  # 增加到达终点的奖励
        
        # 正常移动 - 添加距离奖励
        distance_reward = self._calculate_distance_reward(new_x, new_y)
        self.state = (new_x, new_y, new_vx, new_vy)
        return self.state, -1 + distance_reward, False
    
    def _calculate_distance_reward(self, x: int, y: int) -> float:
        """计算基于距离终点的奖励"""
        if not self.goal_positions:
            return 0.0
        
        # 计算到最近终点的曼哈顿距离
        min_distance = float('inf')
        for goal_x, goal_y in self.goal_positions:
            distance = abs(x - goal_x) + abs(y - goal_y)
            min_distance = min(min_distance, distance)
        
        # 距离越近奖励越高（小幅奖励）
        max_distance = self.track_size[0] + self.track_size[1]
        normalized_distance = min_distance / max_distance
        return 0.1 * (1.0 - normalized_distance)  # 0到0.1的小幅奖励
    
    def _check_collision(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """检查从(x1,y1)到(x2,y2)的路径是否碰撞"""
        # 检查终点是否越界
        if (x2 < 0 or x2 >= self.track_size[0] or 
            y2 < 0 or y2 >= self.track_size[1]):
            return True
        
        # 检查终点是否是墙
        if self.track[x2, y2] == 1:
            return True
        
        # 改进：检查路径上的关键点
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps > 0:
            for i in range(1, steps + 1):
                # 线性插值检查路径
                check_x = int(x1 + (x2 - x1) * i / steps)
                check_y = int(y1 + (y2 - y1) * i / steps)
                
                # 检查是否越界
                if (check_x < 0 or check_x >= self.track_size[0] or 
                    check_y < 0 or check_y >= self.track_size[1]):
                    return True
                
                # 检查是否是墙
                if self.track[check_x, check_y] == 1:
                    return True
        
        return False
    
    def get_state_space_size(self) -> int:
        """获取状态空间大小（用于表格化方法）"""
        return (self.track_size[0] * self.track_size[1] * 
                (2 * self.max_speed + 1) * (2 * self.max_speed + 1))
    
    def state_to_index(self, state: Tuple[int, int, int, int]) -> int:
        """将状态转换为索引（用于表格化方法）"""
        x, y, vx, vy = state
        vx_shifted = vx + self.max_speed
        vy_shifted = vy + self.max_speed
        
        return (x * self.track_size[1] * (2 * self.max_speed + 1) * (2 * self.max_speed + 1) +
                y * (2 * self.max_speed + 1) * (2 * self.max_speed + 1) +
                vx_shifted * (2 * self.max_speed + 1) +
                vy_shifted)
    
    def index_to_state(self, index: int) -> Tuple[int, int, int, int]:
        """将索引转换为状态"""
        speed_range = 2 * self.max_speed + 1
        
        vy_shifted = index % speed_range
        index //= speed_range
        
        vx_shifted = index % speed_range
        index //= speed_range
        
        y = index % self.track_size[1]
        index //= self.track_size[1]
        
        x = index
        
        return (x, y, vx_shifted - self.max_speed, vy_shifted - self.max_speed)
    
    def render(self, show_path: Optional[List] = None):
        """可视化赛道和路径"""
        plt.figure(figsize=(10, 8))
        
        # 创建显示地图
        display_map = self.track.copy().astype(float)
        
        # 颜色映射：0(空地)=白色，1(墙)=黑色，2(起点)=绿色，3(终点)=红色
        colors = ['white', 'black', 'green', 'red']
        
        # 显示当前智能体位置
        if self.state:
            x, y, vx, vy = self.state
            if 0 <= x < self.track_size[0] and 0 <= y < self.track_size[1]:
                if display_map[x, y] != 1:  # 如果不是墙
                    display_map[x, y] = 0.5  # 蓝色表示智能体
        
        plt.imshow(display_map, cmap='tab10', vmin=0, vmax=3)
        
        # 显示路径
        if show_path:
            path_x = [pos[0] for pos in show_path]
            path_y = [pos[1] for pos in show_path]
            plt.plot(path_y, path_x, 'b-', linewidth=2, alpha=0.7)
        
        plt.title('Racetrack Environment')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Terrain')
        plt.show() 