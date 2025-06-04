import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from collections import deque
from simple_racetrack_env import SimpleRacetrackEnv


class CompactActorCriticNetwork(nn.Module):
    """专为小状态空间设计的紧凑Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(CompactActorCriticNetwork, self).__init__()
        
        # 更小更简单的网络架构
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor输出动作概率
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic输出状态价值
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        return F.softmax(action_logits, dim=-1), value


class FastActorCriticAgent:
    """
    专为简化环境设计的快速收敛Actor-Critic智能体
    主要优化：
    1. 适配小状态空间
    2. 更激进的学习率
    3. 更好的奖励塑形
    4. 自适应探索策略
    """
    
    def __init__(self, env: SimpleRacetrackEnv, lr=0.01, gamma=0.95, hidden_dim=64):
        self.env = env
        self.gamma = gamma
        
        # 简化状态特征：位置+速度+目标信息
        self.state_dim = 6
        self.action_dim = env.n_actions
        
        # 创建紧凑网络
        self.network = CompactActorCriticNetwork(self.state_dim, self.action_dim, hidden_dim)
        
        # 使用更高的学习率加速收敛
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 自适应探索
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.losses: List[float] = []
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """简化状态表示，专注于关键特征"""
        x, y, vx, vy = state
        
        # 基础归一化
        norm_x = x / 15.0  # 0-15范围
        norm_y = y / 9.0   # 0-9范围
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 到最近终点的归一化距离
        min_distance = float('inf')
        for goal_x, goal_y in self.env.goal_positions:
            distance = abs(x - goal_x) + abs(y - goal_y)
            min_distance = min(min_distance, distance)
        
        norm_distance = min_distance / 25.0  # 最大可能距离约25
        
        # 简单的位置编码（是否在关键区域）
        in_critical_zone = 1.0 if (x <= 4 or (x >= 12 and y <= 6)) else 0.0
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy, norm_distance, in_critical_zone
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], training=True) -> Tuple[int, torch.Tensor]:
        """使用自适应探索策略选择动作"""
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            action_probs, _ = self.network(state_tensor)
            
            # 应用动作掩码（避免明显的碰撞）
            action_probs = self._apply_action_mask(state, action_probs)
            
            if training and random.random() < self.epsilon:
                # 探索：随机选择
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            else:
                # 利用：贪心选择
                action = torch.argmax(action_probs)
            
            # 计算log概率用于训练
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob
    
    def _apply_action_mask(self, state: Tuple[int, int, int, int], 
                          action_probs: torch.Tensor) -> torch.Tensor:
        """应用动作掩码，避免明显错误的动作"""
        x, y, vx, vy = state
        mask = torch.ones_like(action_probs)
        
        # 检查每个动作是否会导致立即碰撞
        for i, (ax, ay) in enumerate(self.env.actions):
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            new_x = x - new_vx
            new_y = y + new_vy
            
            # 简单碰撞检查
            if (new_x < 0 or new_x >= self.env.track_size[0] or 
                new_y < 0 or new_y >= self.env.track_size[1]):
                mask[i] = 0.1  # 大幅降低但不完全禁止
            elif (0 <= new_x < self.env.track_size[0] and 
                  0 <= new_y < self.env.track_size[1] and 
                  self.env.track[new_x, new_y] == 1):
                mask[i] = 0.1  # 大幅降低但不完全禁止
        
        # 重新归一化
        masked_probs = action_probs * mask
        return masked_probs / (masked_probs.sum() + 1e-8)
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 100  # 小环境用更少步数
        
        # 存储轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        while steps < max_steps:
            # 选择动作
            action, log_prob = self.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            
            # 奖励塑形
            shaped_reward = self._reward_shaping(state, next_state, reward, done, steps)
            
            # 存储经验
            states.append(self.state_to_tensor(state))
            actions.append(action)
            rewards.append(shaped_reward)
            log_probs.append(log_prob)
            
            total_reward += reward  # 原始奖励用于统计
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 计算损失并更新
        if len(states) > 0:
            self._update_network(states, actions, rewards, log_probs)
        
        # 更新探索率
        if episode_num % 50 == 0:  # 每50轮衰减一次
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        success = (done and total_reward > 0)
        return total_reward, steps, success
    
    def _reward_shaping(self, state, next_state, reward, done, steps):
        """针对小环境的奖励塑形"""
        if done and reward > 0:
            # 成功到达，额外奖励与步数成反比
            efficiency_bonus = max(0, (50 - steps) * 0.5)
            return reward + efficiency_bonus
        
        if done and reward < 0:
            return -10  # 碰撞惩罚
        
        # 进步奖励
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        progress_reward = (curr_dist - next_dist) * 2.0
        
        # 接近终点的指数奖励
        proximity_bonus = 0.0
        if next_dist <= 3:
            proximity_bonus = (4 - next_dist) * 1.5
        
        # 减少步数惩罚
        step_penalty = -0.01
        
        return step_penalty + progress_reward + proximity_bonus
    
    def _update_network(self, states, actions, rewards, log_probs):
        """更新网络参数"""
        states = torch.stack(states)
        actions = torch.tensor(actions)
        log_probs = torch.stack(log_probs)
        
        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        # 标准化奖励
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 前向传播
        action_probs, values = self.network(states)
        values = values.squeeze()
        
        # 计算优势
        advantages = discounted_rewards - values.detach()
        
        # 计算损失
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, discounted_rewards)
        total_loss = actor_loss + 0.5 * critic_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        self.losses.append(total_loss.item())
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int], List[float]]:
        """训练智能体"""
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        
        success_window = deque(maxlen=50)  # 较小的窗口
        
        for episode in range(n_episodes):
            reward, steps, success = self.train_episode(episode)
            self.episode_rewards.append(reward)
            self.episode_steps.append(steps)
            
            success_window.append(1 if success else 0)
            current_success_rate = np.mean(success_window)
            self.success_rate.append(current_success_rate)
            
            if verbose and (episode + 1) % 25 == 0:  # 更频繁的输出
                avg_reward = np.mean(self.episode_rewards[-25:])
                avg_steps = np.mean(self.episode_steps[-25:])
                avg_loss = np.mean(self.losses[-25:]) if self.losses else 0
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, "
                      f"平均步数 = {avg_steps:.1f}, 成功率 = {current_success_rate:.3f}, "
                      f"损失 = {avg_loss:.4f}, ε = {self.epsilon:.3f}")
        
        return self.episode_rewards, self.episode_steps, self.success_rate
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """测试一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 100
        
        with torch.no_grad():
            while steps < max_steps:
                action, _ = self.select_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                
                if done:
                    break
                
                state = next_state
        
        success = (done and total_reward > 0)
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success


def main():
    """主函数"""
    print("=== 快速收敛Actor-Critic算法演示（简化环境）===")
    
    # 创建简化环境
    env = SimpleRacetrackEnv()
    env.print_info()
    
    # 创建快速收敛智能体
    print("\n=== 创建快速收敛Actor-Critic智能体 ===")
    agent = FastActorCriticAgent(
        env=env,
        lr=0.01,      # 更高学习率
        gamma=0.95,   # 稍低折扣因子
        hidden_dim=64 # 更小网络
    )
    
    print(f"智能体配置：")
    print(f"  - 学习率: 0.01（高）")
    print(f"  - 网络大小: 64维隐藏层（小）")
    print(f"  - 状态特征: {agent.state_dim}维（简化）")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before, success_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}, 成功 = {success_before}")
    
    # 快速训练
    print(f"\n=== 开始快速训练 ===")
    n_episodes = 500  # 在小环境中应该很快收敛
    rewards, steps, success_rates = agent.train(n_episodes=n_episodes, verbose=True)
    
    # 分析训练结果
    print(f"\n=== 训练结果分析 ===")
    print(f"总训练回合数: {n_episodes}")
    print(f"最终25回合平均奖励: {np.mean(rewards[-25:]):.2f}")
    print(f"最终25回合平均步数: {np.mean(steps[-25:]):.1f}")
    print(f"最终25回合成功率: {np.mean(success_rates[-25:]):.3f}")
    
    # 训练后测试
    print("\n=== 训练后测试 ===")
    test_results = []
    for i in range(10):
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
        print(f"测试 {i+1}: 奖励={reward:.1f}, 步数={steps}, 成功={success}")
    
    avg_test_reward = np.mean([r[0] for r in test_results])
    avg_test_steps = np.mean([r[1] for r in test_results])
    test_success_rate = np.mean([r[2] for r in test_results])
    
    print(f"\n测试结果汇总（10次）:")
    print(f"  - 平均奖励: {avg_test_reward:.2f}")
    print(f"  - 平均步数: {avg_test_steps:.1f}")
    print(f"  - 成功率: {test_success_rate:.3f}")
    
    # 性能提升分析
    print(f"\n=== 性能提升 ===")
    print(f"奖励提升: {avg_test_reward - reward_before:.2f}")
    print(f"步数变化: {avg_test_steps - steps_before:.0f}")
    print(f"成功率提升: {test_success_rate:.3f}")
    
    # 可视化最佳路径
    print("\n=== 展示最佳路径 ===")
    agent.test_episode(render=True)


if __name__ == "__main__":
    main() 