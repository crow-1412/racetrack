import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random


class SimpleRacetrackEnv:
    """
    简化的赛道环境 - 加速收敛版本
    主要简化：
    1. 更小的赛道尺寸 (16x10)
    2. 更低的最大速度 (3)
    3. 更简单的赛道布局
    4. 状态空间从65,824降低到约3,840
    """
    
    def __init__(self, track_size=(20, 10), max_speed=2):
        self.track_size = track_size
        self.max_speed = max_speed
        self.min_speed = 0
        
        # 创建简化赛道地图 (0:空地, 1:墙, 2:起点, 3:终点)
        self.track = self._create_simple_track()
        self.start_positions = self._get_start_positions()
        self.goal_positions = self._get_goal_positions()
        
        # 动作空间: 9个动作 (-1,-1), (-1,0), ..., (1,1)
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.n_actions = len(self.actions)
        
        # 当前状态
        self.state = None
        self.reset()
    
    def _create_simple_track(self) -> np.ndarray:
        """创建改进的L型赛道 - 更还原原始设计"""
        track = np.ones(self.track_size, dtype=int)  # 默认全是墙
        
        # 更还原原始赛道的L型设计
        # 底部垂直段 - 从起点向上（对应原始赛道的第3-8列）
        track[12:20, 0:4] = 0   # 第13-20行，第1-4列 (垂直通道)
        
        # 中间转弯段（对应原始赛道的第9列过渡区域）
        track[8:13, 2:6] = 0    # 第9-13行，第3-6列 (转弯区域)
        
        # 拐弯后的中间直行段（对应原始赛道第10-15列的部分）
        track[0:9, 4:7] = 0     # 第1-9行，第5-7列 (中间直行段)
        
        # 最终到达终点的水平段（对应原始赛道第16列）
        track[0:6, 7:10] = 0    # 第1-6行，第8-10列 (最终水平段)
        
        # 起点：最后一行的前5个格子（索引0-4）
        track[19, :] = 1        # 先将第19行全部设为墙壁
        track[19, 0:5] = 2      # 起点区域：5个格子
        
        # 终点：最后一列的前5个格子（索引0-4）
        track[:, 9] = 1         # 先将第9列全部设为墙壁
        track[0:5, 9] = 3       # 终点区域：5个格子
        
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
        
        # 0.1概率速度保持不变
        if random.random() < 0.1:
            ax, ay = 0, 0
        
        # 更新速度（严格非负）
        # vx > 0: 向上移动, vy > 0: 向右移动  
        new_vx = max(0, min(self.max_speed, vx + ax))
        new_vy = max(0, min(self.max_speed, vy + ay))
        
        # 速度不能同时为零（除了在起点）
        if new_vx == 0 and new_vy == 0 and (x, y) not in self.start_positions:
            new_vx = 1  # 设置最小速度
            new_vy = 1
        
        # 计算新位置
        # vx > 0 表示向上移动（x坐标减小）
        # vy > 0 表示向右移动（y坐标增大）
        new_x = x - new_vx  # 向上移动时x减小
        new_y = y + new_vy  # 向右移动时y增大
        
        # 检查碰撞
        if self._check_collision(x, y, new_x, new_y):
            # 碰撞：重置到起点
            self.state = self.reset()
            return self.state, -5, False  # 减少碰撞惩罚
        
        # 检查是否到达终点
        if (new_x, new_y) in self.goal_positions:
            self.state = (new_x, new_y, new_vx, new_vy)
            return self.state, 50, True  # 成功奖励
        
        # 正常移动 - 添加距离奖励
        distance_reward = self._calculate_distance_reward(new_x, new_y)
        self.state = (new_x, new_y, new_vx, new_vy)
        return self.state, -0.1 + distance_reward, False  # 更小的步数惩罚
    
    def _calculate_distance_reward(self, x: int, y: int) -> float:
        """计算基于距离终点的奖励"""
        if not self.goal_positions:
            return 0.0
        
        # 计算到最近终点的曼哈顿距离
        min_distance = float('inf')
        for goal_x, goal_y in self.goal_positions:
            distance = abs(x - goal_x) + abs(y - goal_y)
            min_distance = min(min_distance, distance)
        
        # 距离越近奖励越高
        max_distance = self.track_size[0] + self.track_size[1]
        normalized_distance = min_distance / max_distance
        return 0.2 * (1.0 - normalized_distance)  # 0到0.2的奖励
    
    def _check_collision(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """检查从(x1,y1)到(x2,y2)的路径是否碰撞"""
        # 检查终点是否越界
        if (x2 < 0 or x2 >= self.track_size[0] or 
            y2 < 0 or y2 >= self.track_size[1]):
            return True
        
        # 检查终点是否是墙
        if self.track[x2, y2] == 1:
            return True
        
        # 检查路径上的关键点
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
        plt.figure(figsize=(8, 6))
        
        # 创建显示地图
        display_map = self.track.copy().astype(float)
        
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
        
        plt.title('简化赛道环境 (16x10)')
        plt.xlabel('Y坐标')
        plt.ylabel('X坐标')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='地形类型')
        plt.show()
    
    def print_info(self):
        """打印环境信息"""
        state_space_size = self.get_state_space_size()
        print(f"改进的简化赛道环境信息：")
        print(f"  - 赛道大小: {self.track_size}")
        print(f"  - 最大速度: {self.max_speed}")
        print(f"  - 状态空间大小: {state_space_size:,}")
        print(f"  - 起点数量: {len(self.start_positions)}")
        print(f"  - 终点数量: {len(self.goal_positions)}")
        print(f"  - 起点位置: {self.start_positions}")
        print(f"  - 终点位置: {self.goal_positions}")
        
        # 计算与原始环境的比较
        original_size = 32 * 17 * 11 * 11  # 65,824
        reduction_ratio = original_size / state_space_size
        print(f"  - 相比原始环境状态空间减少: {reduction_ratio:.1f}倍")
        print(f"  - 相比第一版简化环境的改进：")
        print(f"    * 赛道更宽敞 (从16x10改为20x10)")
        print(f"    * 更合理的L型布局")
        print(f"    * 墙壁更少，通行性更好")


def demo_track_visualization():
    """专门演示改进的赛道布局"""
    print("=== 改进的赛道布局演示 ===")
    
    env = SimpleRacetrackEnv()
    env.print_info()
    
    print("\n赛道布局矩阵:")
    print("  0=道路, 1=墙壁, 2=起点, 3=终点")
    print("=" * 50)
    
    # 打印矩阵，每行显示行号
    for i, row in enumerate(env.track):
        row_str = " ".join(str(cell) for cell in row)
        print(f"行{i:2d}: {row_str}")
    
    print("=" * 50)
    print(f"赛道特点总结:")
    print(f"  - 起点位置: {env.start_positions}")
    print(f"  - 终点位置: {env.goal_positions}")
    print(f"  - L型赛道，从底部起点向上再向右到达终点")
    print(f"  - 布局更紧凑，去除了不必要的空隙")


if __name__ == "__main__":
    demo_track_visualization() 