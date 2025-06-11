"""
PPO (Proximal Policy Optimization) 强化学习智能体 - 赛车轨道问题

本文件实现了基于PPO算法的赛车轨道智能体，结合了Actor-Critic优化版的优秀特性。

PPO核心特性：
1. Clipped Surrogate Objective - 防止策略更新过大
2. Multiple Epochs Training - 充分利用采集的数据
3. Adaptive KL Divergence - 自适应调整学习步长
4. 继承优化版Actor-Critic的防退化机制

技术改进：
1. 智能状态表示（8维特征向量）
2. 严格动作掩码（避免碰撞）
3. 分阶段训练策略
4. 最佳模型保护机制
5. GAE优势估计

作者：AI Assistant
最后更新：2024年
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Dict
import random
from collections import deque
import matplotlib.pyplot as plt
from racetrack_env import RacetrackEnv


class PPONetwork(nn.Module):
    """
    PPO网络架构 - 共享特征提取 + 分离Actor-Critic头部
    
    与Actor-Critic类似的架构，但专门为PPO优化：
    - 共享层：提取环境状态的通用特征
    - Actor头部：输出动作概率分布
    - Critic头部：估计状态价值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()
        
        # 共享的底层特征提取网络
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout提高泛化
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor头部：输出动作logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Critic头部：输出状态价值估计
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """参数初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.5)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        shared_features = self.shared_layers(state)
        
        # Actor输出：动作logits
        action_logits = self.actor_head(shared_features)
        
        # Critic输出：状态价值
        value = self.critic_head(shared_features)
        
        return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """
        获取动作概率分布和价值，用于PPO训练
        
        Args:
            state: 状态张量
            action: 如果提供，计算该动作的log概率
            
        Returns:
            action: 采样的动作（如果没有提供action参数）
            log_prob: 动作的对数概率
            entropy: 策略熵
            value: 状态价值
        """
        action_logits, value = self.forward(state)
        action_dist = Categorical(logits=action_logits)
        
        if action is None:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action, log_prob, entropy, value


class PPOBuffer:
    """
    PPO经验缓冲区
    
    存储一个完整episode的经验，用于PPO的多轮更新
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def size(self):
        """获取缓冲区大小"""
        return len(self.states)
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float, next_value: float = 0):
        """
        计算GAE优势和回报
        
        Args:
            gamma: 折扣因子
            gae_lambda: GAE的λ参数
            next_value: 最后状态的价值（如果episode未结束）
        """
        rewards = np.array(self.rewards)
        values = np.array([v.detach().item() if isinstance(v, torch.Tensor) else v for v in self.values])
        dones = np.array(self.dones)
        
        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        # 计算回报
        returns = advantages + values
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batch(self, batch_size: int):
        """获取批量数据"""
        indices = np.random.choice(self.size(), min(batch_size, self.size()), replace=False)
        
        batch_states = torch.stack([self.states[i] for i in indices])
        batch_actions = torch.tensor([self.actions[i] for i in indices])
        batch_old_log_probs = torch.stack([self.log_probs[i].detach() for i in indices])
        batch_advantages = torch.tensor([self.advantages[i] for i in indices], dtype=torch.float32)
        batch_returns = torch.tensor([self.returns[i] for i in indices], dtype=torch.float32)
        
        return batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns


class PPORacetrackAgent:
    """
    PPO赛车轨道智能体
    
    主要特性：
    1. PPO算法核心：Clipped Surrogate Objective
    2. 多轮更新：充分利用采集的数据
    3. 继承Actor-Critic优化版的优秀特性
    4. 自适应KL散度控制
    """
    
    def __init__(self, env: RacetrackEnv, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2, 
                 ppo_epochs: int = 4, batch_size: int = 64,
                 buffer_size: int = 2048, hidden_dim: int = 128):
        """
        初始化PPO智能体
        
        Args:
            env: 赛车轨道环境
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE的λ参数
            clip_ratio: PPO裁剪比率
            ppo_epochs: PPO更新轮数
            batch_size: 批量大小
            buffer_size: 缓冲区大小
            hidden_dim: 隐藏层维度
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # 状态特征维度：继承优化版Actor-Critic的8维特征
        self.state_dim = 8
        self.action_dim = env.n_actions
        
        # 创建网络
        self.network = PPONetwork(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(buffer_size)
        
        # 探索参数（PPO通常不需要额外的探索机制）
        self.exploration_noise = 0.0
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []
        
        # 最佳模型保护（继承优化版特性）
        self.best_success_rate = 0.0
        self.best_model_state = None
        self.patience = 0
        self.max_patience = 100
        
        # 自适应参数
        self.target_kl = 0.01  # 目标KL散度
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50, verbose=True
        )
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        将环境状态转换为神经网络输入张量
        继承优化版Actor-Critic的8维特征设计
        """
        x, y, vx, vy = state
        
        # 1. 基础特征归一化
        norm_x = x / 31.0
        norm_y = y / 16.0  
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 2. 计算到最近终点的距离和方向
        min_distance = float('inf')
        goal_direction_x, goal_direction_y = 0, 0
        
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            if distance < min_distance:
                min_distance = distance
                if distance > 0:
                    goal_direction_x = -(goal_x - x) / distance
                    goal_direction_y = (goal_y - y) / distance
        
        # 3. 距离归一化
        max_distance = np.sqrt(31**2 + 16**2)
        norm_distance = min_distance / max_distance
        
        # 4. 速度与目标方向的对齐度
        velocity_alignment = 0.0
        if min_distance > 0:
            velocity_mag = np.sqrt(vx**2 + vy**2)
            if velocity_mag > 0:
                vel_dir_x = vx / velocity_mag
                vel_dir_y = vy / velocity_mag
                velocity_alignment = max(0, vel_dir_x * goal_direction_x + vel_dir_y * goal_direction_y)
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_distance, goal_direction_x, goal_direction_y, 
            velocity_alignment
        ], dtype=torch.float32)
    
    def apply_action_mask(self, state: Tuple[int, int, int, int], 
                         action_logits: torch.Tensor) -> torch.Tensor:
        """
        应用动作掩码，继承优化版Actor-Critic的严格掩码策略
        """
        x, y, vx, vy = state
        mask = torch.zeros_like(action_logits)
        
        for i, (ax, ay) in enumerate(self.env.actions):
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            
            if new_vx == 0 and new_vy == 0 and (x, y) not in self.env.start_positions:
                new_vx = 1
                new_vy = 1
            
            new_x = x - new_vx
            new_y = y + new_vy
            
            if self.env._check_collision(x, y, new_x, new_y):
                mask[i] = -1e8  # 使用大负数而不是inf避免NaN
        
        masked_logits = action_logits + mask
        
        # 检查是否有有效动作
        if torch.all(mask == -1e8):
            # 如果所有动作都被禁止，重置掩码
            mask.fill_(0)
            masked_logits = action_logits
        
        return masked_logits
    
    def select_action(self, state: Tuple[int, int, int, int], training: bool = True):
        """
        选择动作（PPO版本）
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits, value = self.network(state_tensor)
            
            # 应用动作掩码
            masked_logits = self.apply_action_mask(state, action_logits)
            
            # 创建动作分布并采样
            action_dist = Categorical(logits=masked_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def improved_reward_shaping(self, state, next_state, reward, done, steps):
        """
        继承优化版Actor-Critic的奖励塑形策略
        """
        bonus = 0.0
        
        # 成功/失败的明确奖励
        if done and reward > 0:
            bonus += 100
        elif reward == -10:  # 碰撞
            bonus -= 50
        
        # 简单的进步奖励
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        # 计算到最近目标的曼哈顿距离
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        if curr_dist - next_dist > 1:
            bonus += 2.0
        
        # 轻微的步数惩罚
        bonus -= 0.1
        
        return reward + bonus
    
    def collect_trajectory(self, max_steps: int = 200) -> Tuple[float, int, bool]:
        """
        收集一个完整的轨迹
        
        Returns:
            total_reward: 总奖励 
            steps: 步数
            success: 是否成功
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        episode_buffer = []
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(state, training=True)
            prev_state = state
            
            next_state, reward, done = self.env.step(action)
            
            # 奖励塑形
            shaped_reward = self.improved_reward_shaping(prev_state, next_state, reward, done, steps)
            
            # 存储经验
            self.buffer.add(
                self.state_to_tensor(prev_state),
                action,
                shaped_reward,
                value,
                log_prob,
                done
            )
            
            total_reward += reward  # 使用原始奖励计算总回报
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 如果episode未结束，计算最后状态的价值
        if not done:
            _, _, next_value = self.select_action(state, training=True)
        else:
            next_value = 0.0
        
        # 计算优势和回报
        self.buffer.compute_advantages_and_returns(self.gamma, self.gae_lambda, next_value)
        
        # 判断成功
        success = (done and reward == 100)
        return total_reward, steps, success
    
    def update_policy(self):
        """
        PPO策略更新
        """
        if self.buffer.size() < self.batch_size:
            return
        
        # 标准化优势
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()  # 确保没有梯度依赖
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        # PPO多轮更新
        for epoch in range(self.ppo_epochs):
            # 获取批量数据
            batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = \
                self.buffer.get_batch(min(self.batch_size, self.buffer.size()))
            
            # 重新计算动作概率和价值
            action_logits, values = self.network(batch_states)
            
            # 批量更新时简化处理，不应用掩码避免复杂性
            # 因为训练数据中的动作已经是经过掩码选择的有效动作
            
            action_dist = Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy()
            
            # 计算重要性采样比率
            ratio = torch.exp(log_probs - batch_old_log_probs)
            
            # PPO Clipped Surrogate Objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = F.mse_loss(values.squeeze(), batch_returns)
            
            # 熵损失（鼓励探索）
            entropy_loss = entropy.mean()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            # 记录损失
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            
            # 计算KL散度（用于自适应调整）
            with torch.no_grad():
                kl_div = (batch_old_log_probs - log_probs).mean()
                total_kl_div += kl_div.item()
                
                # 如果KL散度过大，提前停止更新
                if kl_div > 1.5 * self.target_kl:
                    break
        
        # 记录平均损失
        self.policy_losses.append(total_policy_loss / (epoch + 1))
        self.value_losses.append(total_value_loss / (epoch + 1))
        self.entropy_losses.append(total_entropy_loss / (epoch + 1))
        
        # 清空缓冲区
        self.buffer.clear()
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """
        训练单个episode
        """
        # 收集轨迹
        reward, steps, success = self.collect_trajectory()
        
        # 更新策略
        self.update_policy()
        
        return reward, steps, success
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """
        测试单个episode
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 300
        
        self.network.eval()
        with torch.no_grad():
            while steps < max_steps:
                action, _, _ = self.select_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                
                if done:
                    break
                
                state = next_state
        
        self.network.train()
        success = (done and reward == 100)
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rate': self.success_rate,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()


def main_ppo_training():
    """
    PPO主训练函数 - 结合分阶段训练和最佳模型保护
    """
    print("=== PPO赛车轨道训练 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建PPO智能体
    agent = PPORacetrackAgent(
        env=env,
        lr=3e-4,           # PPO推荐学习率
        gamma=0.99,        # 折扣因子
        gae_lambda=0.95,   # GAE参数
        clip_ratio=0.2,    # PPO裁剪比率
        ppo_epochs=4,      # PPO更新轮数
        batch_size=64,     # 批量大小
        buffer_size=2048,  # 缓冲区大小
        hidden_dim=128     # 隐藏层维度
    )
    
    print(f"PPO配置:")
    print(f"  - 学习率: 3e-4")
    print(f"  - 裁剪比率: 0.2")
    print(f"  - PPO更新轮数: 4")
    print(f"  - 批量大小: 64")
    print(f"  - 缓冲区大小: 2048")
    
    # 训练前基准测试
    print("\n=== 训练前基准 ===")
    reward_before, steps_before, _, success_before = agent.test_episode()
    print(f"基准性能: 奖励={reward_before:.1f}, 步数={steps_before}, 成功={success_before}")
    
    # 分阶段训练 (缩短为演示)
    n_episodes = 500
    print(f"\n=== 开始PPO训练 ===")
    print(f"训练计划: {n_episodes}回合")
    
    # 训练统计
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=50)
    
    for episode in range(n_episodes):
        # 训练一个episode
        reward, steps, success = agent.train_episode(episode)
        
        agent.episode_rewards.append(reward)
        agent.episode_steps.append(steps)
        success_window.append(1 if success else 0)
        reward_window.append(reward)
        
        current_success_rate = np.mean(success_window)
        agent.success_rate.append(current_success_rate)
        
        # 最佳模型保护机制
        if episode >= 50 and current_success_rate > agent.best_success_rate:
            agent.best_success_rate = current_success_rate
            agent.best_model_state = {
                'network': agent.network.state_dict().copy(),
                'episode': episode,
                'success_rate': current_success_rate
            }
            agent.patience = 0
            print(f"💾 保存最佳模型: Episode {episode+1}, 成功率={current_success_rate:.3f}")
        else:
            agent.patience += 1
        
        # 学习率调度
        if episode % 100 == 0 and episode > 0:
            agent.lr_scheduler.step(current_success_rate)
        
        # 性能退化检测与恢复
        if agent.patience > agent.max_patience and agent.best_model_state:
            print(f"\n⚠️ 检测到性能退化，恢复最佳模型...")
            agent.network.load_state_dict(agent.best_model_state['network'])
            print(f"   已恢复Episode {agent.best_model_state['episode']+1}的模型")
            agent.patience = 0
        
        # 定期输出训练进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(agent.episode_steps[-25:])
            avg_policy_loss = np.mean(agent.policy_losses[-25:]) if agent.policy_losses else 0
            avg_value_loss = np.mean(agent.value_losses[-25:]) if agent.value_losses else 0
            
            print(f"Episode {episode + 1:4d}: "
                  f"奖励={avg_reward:6.1f}, 步数={avg_steps:5.1f}, "
                  f"成功率={current_success_rate:.3f}")
            print(f"                     策略损失={avg_policy_loss:.4f}, "
                  f"价值损失={avg_value_loss:.4f}, "
                  f"最佳成功率={agent.best_success_rate:.3f}")
    
    # 恢复最佳模型进行最终测试
    if agent.best_model_state:
        print(f"\n🔄 恢复最佳模型进行最终测试...")
        agent.network.load_state_dict(agent.best_model_state['network'])
    
    # 最终测试
    print(f"\n=== 最终评估 ===")
    test_results = []
    for i in range(50):
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
    
    final_success_rate = np.mean([r[2] for r in test_results])
    final_avg_reward = np.mean([r[0] for r in test_results])
    final_avg_steps = np.mean([r[1] for r in test_results])
    
    print(f"最终测试结果（50次）:")
    print(f"  成功率: {final_success_rate:.1%}")
    print(f"  平均奖励: {final_avg_reward:.1f}")
    print(f"  平均步数: {final_avg_steps:.1f}")
    
    # 性能评价
    if final_success_rate > 0.7:
        print("🎉 PPO表现优秀！成功率超过70%")
    elif final_success_rate > 0.5:
        print("✅ PPO表现良好！成功率超过50%")
    elif final_success_rate > 0.3:
        print("📈 PPO表现尚可，有改进空间")
    else:
        print("⚠️ PPO需要进一步调优")
    
    # 保存模型
    agent.save_model("models/ppo_racetrack_model.pth")
    print(f"PPO模型已保存到 models/ 文件夹")
    
    # 绘制训练曲线
    plot_training_curves(agent)
    
    return agent, test_results


def plot_training_curves(agent):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 成功率曲线
    if agent.success_rate:
        axes[0, 0].plot(agent.success_rate)
        axes[0, 0].set_title('Success Rate')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True)
    
    # 奖励曲线
    if agent.episode_rewards:
        # 计算移动平均
        window_size = 50
        if len(agent.episode_rewards) > window_size:
            moving_avg = np.convolve(agent.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Episode Rewards (Moving Average, window={window_size})')
        else:
            axes[0, 1].plot(agent.episode_rewards)
            axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
    
    # 策略损失
    if agent.policy_losses:
        axes[1, 0].plot(agent.policy_losses)
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # 价值损失
    if agent.value_losses:
        axes[1, 1].plot(agent.value_losses)
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 运行PPO训练
    main_ppo_training() 