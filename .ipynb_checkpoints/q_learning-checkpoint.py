import numpy as np
from typing import Tuple, List, Dict
import random
from racetrack_env import RacetrackEnv


class QLearningAgent:
    """
    Q-learning 智能体
    离策略时序差分学习
    """
    
    def __init__(self, env: RacetrackEnv, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化Q表
        self.state_space_size = env.get_state_space_size()
        self.n_actions = env.n_actions
        
        # 使用字典存储Q值（稀疏表示）
        self.Q: Dict[Tuple[Tuple[int, int, int, int], int], float] = {}
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
    
    def get_q_value(self, state: Tuple[int, int, int, int], action: int) -> float:
        """获取Q值"""
        state_key = state
        if (state_key, action) not in self.Q:
            self.Q[(state_key, action)] = 0.0
        return self.Q[(state_key, action)]
    
    def set_q_value(self, state: Tuple[int, int, int, int], action: int, value: float):
        """设置Q值"""
        state_key = state
        self.Q[(state_key, action)] = value
    
    def get_max_q_value(self, state: Tuple[int, int, int, int]) -> float:
        """获取状态的最大Q值"""
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return max(q_values)
    
    def epsilon_greedy_action(self, state: Tuple[int, int, int, int]) -> int:
        """ε-greedy 动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # 贪婪选择
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            max_q = max(q_values)
            # 如果有多个最大值，随机选择一个
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个episode"""
        state = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        
        while True:
            # 选择动作
            action = self.epsilon_greedy_action(state)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # Q-learning 更新（使用最大Q值）
            q_current = self.get_q_value(state, action)
            q_max_next = self.get_max_q_value(next_state) if not done else 0.0
            
            # TD误差
            delta = reward + self.gamma * q_max_next - q_current
            
            # 更新Q值
            new_q = q_current + self.alpha * delta
            self.set_q_value(state, action, new_q)
            
            if done:
                break
            
            state = next_state
        
        return total_reward, steps
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int]]:
        """训练智能体"""
        self.episode_rewards = []
        self.episode_steps = []
        
        for episode in range(n_episodes):
            reward, steps = self.train_episode()
            self.episode_rewards.append(reward)
            self.episode_steps.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, 平均步数 = {avg_steps:.2f}")
        
        return self.episode_rewards, self.episode_steps
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List]:
        """测试一个episode（使用贪婪策略）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]  # 记录路径（只记录位置）
        
        while steps < 1000:  # 防止无限循环
            # 贪婪选择动作
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            action = int(np.argmax(q_values))
            
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            path.append(next_state[:2])
            
            if done:
                break
            
            state = next_state
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path
    
    def get_policy(self) -> Dict[Tuple[int, int, int, int], int]:
        """获取当前策略（贪婪策略）"""
        policy: Dict[Tuple[int, int, int, int], int] = {}
        for (state, action), q_value in self.Q.items():
            if state not in policy:
                policy[state] = action
            else:
                current_q = self.get_q_value(state, policy[state])
                if q_value > current_q:
                    policy[state] = action
        return policy 