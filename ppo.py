"""
PPO (Proximal Policy Optimization) 强化学习智能体 - 赛车轨道问题 - 优化版

优化改进：
1. 智能奖励塑形 - 对前进给予微弱正奖励，解决稀疏奖励问题
2. 放松KL散度限制 - 允许更大的策略更新
3. 调整学习率 - 提高到合理水平
4. 增大批量大小 - 提高训练稳定性
5. 改进网络架构 - 更适合离散动作空间

作者：AI Assistant  
最后更新：2024年 - 优化版
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

# 设置随机种子确保结果可重现
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"🎲 优化版PPO随机种子已设置为: {RANDOM_SEED}")


class OptimizedPPONetwork(nn.Module):
    """
    优化版PPO网络架构
    
    改进：
    - 适中的网络规模
    - 更好的初始化
    - 针对离散动作空间优化
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(OptimizedPPONetwork, self).__init__()
        
        # 适中的网络结构
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor头部
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic头部  
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
        # 合理的参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """合理的参数初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        return action_logits, value


class OptimizedPPOBuffer:
    """
    优化版PPO经验缓冲区
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
        优化的GAE优势计算
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array([v.detach().item() if isinstance(v, torch.Tensor) else v for v in self.values], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)
        
        # 合理的奖励裁剪
        rewards = np.clip(rewards, -50, 50)
        
        # GAE计算
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        # 计算returns
        returns = advantages + values
        
        # 数值稳定性检查
        if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
            print("⚠️ 警告：GAE计算异常，使用简单TD误差")
            advantages = rewards - values
            returns = rewards.copy()
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batch(self, batch_size: int):
        """获取批量数据"""
        indices = np.random.choice(self.size(), min(batch_size, self.size()), replace=False)
        
        batch_states = torch.stack([self.states[i] for i in indices])
        batch_actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.long)
        batch_old_log_probs = torch.stack([self.log_probs[i].detach() for i in indices])
        batch_advantages = torch.tensor([self.advantages[i] for i in indices], dtype=torch.float32)
        batch_returns = torch.tensor([self.returns[i] for i in indices], dtype=torch.float32)
        
        return batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns


class OptimizedPPORacetrackAgent:
    """
    优化版PPO赛车轨道智能体
    
    主要优化：
    1. 智能奖励塑形
    2. 放松KL散度限制
    3. 合理的学习率和批量大小
    4. 改进的网络架构
    """
    
    def __init__(self, env: RacetrackEnv, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,  # 标准裁剪比例
                 ppo_epochs: int = 4, batch_size: int = 128,  # 增大批量和更新轮数
                 buffer_size: int = 1024, hidden_dim: int = 128):  # 增大缓冲区和网络
        """
        初始化优化版PPO智能体
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # 状态特征维度
        self.state_dim = 8
        self.action_dim = env.n_actions
        
        # 创建优化版网络
        self.network = OptimizedPPONetwork(self.state_dim, self.action_dim, hidden_dim)
        
        # 合理的学习率
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=3e-4,  # 标准学习率
            eps=1e-5
        )
        
        # 优化版缓冲区
        self.buffer = OptimizedPPOBuffer(buffer_size)
        
        # 合理的探索策略
        self.epsilon = 0.05  # 🔥 从0.1降到0.05，减少随机探索
        self.epsilon_min = 0.01  # 从0.02降到0.01
        self.epsilon_decay = 0.995
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []
        self.kl_divergences: List[float] = []
        
        # 放松的超参数
        self.target_kl = 0.5  # 放松KL散度限制
        self.value_coef = 0.5  # 标准价值损失权重
        self.entropy_coef = 0.01  # 适中的熵系数
        
        # 最佳模型保护
        self.best_success_rate = 0.0
        self.best_model_state = None
        self.patience = 0
        self.max_patience = 100
        
        # 奖励塑形参数
        self.last_distance_to_goal = None
        self.progress_reward_scale = 0.1  # 前进奖励的缩放因子
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """状态转换为张量"""
        x, y, vx, vy = state
        
        # 基础特征归一化
        norm_x = x / 31.0
        norm_y = y / 16.0  
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 计算到最近终点的距离和方向
        min_distance = float('inf')
        goal_direction_x, goal_direction_y = 0, 0
        
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            if distance < min_distance:
                min_distance = distance
                if distance > 0:
                    goal_direction_x = -(goal_x - x) / distance
                    goal_direction_y = (goal_y - y) / distance
        
        # 距离归一化
        max_distance = np.sqrt(31**2 + 16**2)
        norm_distance = min_distance / max_distance
        
        # 速度与目标方向的对齐度
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
        """应用动作掩码"""
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
                mask[i] = -1e9  # 大的负值掩码
        
        masked_logits = action_logits + mask
        
        if torch.all(mask == -1e9):
            mask.fill_(0)
            masked_logits = action_logits
        
        return masked_logits
    
    def select_action(self, state: Tuple[int, int, int, int], training: bool = True):
        """选择动作（修复版 - 彻底解决训练测试不一致问题）"""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits, value = self.network(state_tensor)
            
            # 应用动作掩码
            masked_logits = self.apply_action_mask(state, action_logits)
            
            # 创建动作分布
            action_dist = Categorical(logits=masked_logits)
            
            # 🔥 关键修复：统一动作选择逻辑
            if training:
                # 训练时：从分布中采样（带有一定随机性）
                if random.random() < self.epsilon:
                    # 温和的探索：从softmax分布采样而不是完全随机
                    action = action_dist.sample()
                else:
                    # 大部分时候仍使用贪婪策略，确保网络学习正确方向
                    action = torch.argmax(masked_logits)
            else:
                # 测试时：也使用相同的贪婪策略
                action = torch.argmax(masked_logits)
            
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def intelligent_reward_shaping(self, prev_state, state, next_state, reward, done, steps):
        """
        修正版奖励塑形 - 减少过度乐观，确保网络学到真实策略
        """
        shaped_reward = reward  # 保留原始奖励
        
        x, y, vx, vy = state
        
        # 1. 前进奖励 - 大幅减弱，避免虚假信号
        current_distance = float('inf')
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            current_distance = min(current_distance, distance)
        
        if self.last_distance_to_goal is not None:
            progress = self.last_distance_to_goal - current_distance
            if progress > 0:  # 向目标靠近
                # 🔥 关键修复：大幅减少前进奖励，避免虚假积极信号
                shaped_reward += progress * 0.02  # 从0.1降到0.02
            elif progress < -2:  # 远离目标太多
                shaped_reward -= 0.02  # 轻微惩罚
        
        self.last_distance_to_goal = current_distance
        
        # 2. 速度奖励 - 减弱
        speed = np.sqrt(vx**2 + vy**2)
        if 1 <= speed <= 3:  # 合理速度范围
            shaped_reward += 0.005  # 从0.02降到0.005
        elif speed == 0:  # 惩罚停止
            shaped_reward -= 0.02
        
        # 3. 方向奖励 - 减弱但保留
        if current_distance > 0:
            goal_direction_x = -(self.env.goal_positions[0][0] - x) / current_distance
            goal_direction_y = (self.env.goal_positions[0][1] - y) / current_distance
            
            if speed > 0:
                vel_dir_x = vx / speed
                vel_dir_y = vy / speed
                alignment = vel_dir_x * goal_direction_x + vel_dir_y * goal_direction_y
                if alignment > 0.5:  # 方向对齐
                    shaped_reward += 0.005  # 从0.01降到0.005
        
        # 4. 步数惩罚 - 保持
        shaped_reward -= 0.01
        
        # 5. 特殊情况处理 - 强化真实成功奖励
        if done:
            if reward == 100:  # 成功
                shaped_reward += 50  # 大幅奖励真实成功
            elif reward == -10:  # 碰撞
                shaped_reward -= 10   # 增加碰撞惩罚
            else:  # 超时
                shaped_reward -= 5    # 增加超时惩罚
        
        return shaped_reward
    
    def collect_trajectory(self, max_steps: int = 300) -> Tuple[float, int, bool]:
        """收集轨迹"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        self.last_distance_to_goal = None  # 重置距离跟踪
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(state, training=True)
            prev_state = state
            
            next_state, reward, done = self.env.step(action)
            
            # 智能奖励塑形
            shaped_reward = self.intelligent_reward_shaping(
                prev_state, state, next_state, reward, done, steps
            )
            
            # 存储经验
            self.buffer.add(
                self.state_to_tensor(prev_state),
                action,
                shaped_reward,
                value,
                log_prob,
                done
            )
            
            total_reward += reward  # 记录原始奖励
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 计算最后状态的价值
        if not done:
            _, _, next_value = self.select_action(state, training=True)
        else:
            next_value = 0.0
        
        # 计算优势和回报
        self.buffer.compute_advantages_and_returns(self.gamma, self.gae_lambda, next_value)
        
        success = (done and reward == 100)
        return total_reward, steps, success
    
    def update_policy(self):
        """
        优化的PPO策略更新
        """
        if self.buffer.size() < self.batch_size:
            return
        
        # 优势归一化
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        
        # 检查数值稳定性
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("⚠️ 优势计算异常，跳过此次更新")
            self.buffer.clear()
            return
        
        # 标准优势归一化
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        update_count = 0
        
        # PPO更新
        for epoch in range(self.ppo_epochs):
            # 获取批量数据
            batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = \
                self.buffer.get_batch(self.batch_size)
            
            # 使用归一化的优势
            batch_advantages = advantages[:len(batch_advantages)]
            
            # 重新计算动作概率和价值
            action_logits, values = self.network(batch_states)
            
            action_dist = Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy()
            
            # 计算重要性采样比率
            ratio = torch.exp(log_probs - batch_old_log_probs)
            
            # 检查比率是否异常
            if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                print("⚠️ 重要性采样比率异常，跳过此次更新")
                continue
            
            # PPO Clipped Surrogate Objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            batch_returns_tensor = torch.tensor(self.buffer.returns[:len(batch_returns)], dtype=torch.float32)
            value_loss = F.mse_loss(values.squeeze(), batch_returns_tensor)
            
            # 熵损失
            entropy_loss = entropy.mean()
            
            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # 检查损失是否异常
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("⚠️ 损失函数异常，跳过此次更新")
                continue
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            
            self.optimizer.step()
            
            # 记录损失
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            update_count += 1
            
            # 计算KL散度（放松检查）
            with torch.no_grad():
                kl_div = (batch_old_log_probs - log_probs).mean()
                total_kl_div += kl_div.item()
                
                # 放松的KL散度控制
                if kl_div > 2.0 * self.target_kl:  # 更宽松的限制
                    print(f"📊 KL散度较大 ({kl_div:.4f}), 提前停止epoch {epoch}")
                    break
        
        # 记录平均损失
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy_loss = total_entropy_loss / update_count
            avg_kl_div = total_kl_div / update_count
            
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            self.entropy_losses.append(avg_entropy_loss)
            self.kl_divergences.append(avg_kl_div)
        
        # 清空缓冲区
        self.buffer.clear()
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练单个episode"""
        # 收集轨迹
        reward, steps, success = self.collect_trajectory()
        
        # 更新策略
        self.update_policy()
        
        # 探索率衰减
        if episode_num % 10 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return reward, steps, success
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """测试单个episode（修复版 - 确保与训练逻辑一致）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 300
        
        # 重置奖励塑形状态
        self.last_distance_to_goal = None
        
        self.network.eval()
        with torch.no_grad():
            while steps < max_steps:
                action, _, _ = self.select_action(state, training=False)
                prev_state = state
                next_state, reward, done = self.env.step(action)
                
                # 🔥 关键修复：测试时也使用奖励塑形逻辑（但只用于成功判断）
                shaped_reward = self.intelligent_reward_shaping(
                    prev_state, state, next_state, reward, done, steps
                )
                
                total_reward += reward  # 记录原始奖励
                steps += 1
                path.append(next_state[:2])
                
                if done:
                    break
                
                state = next_state
        
        self.network.train()
        
        # 使用原始奖励进行成功判断（与训练时一致）
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
            'entropy_losses': self.entropy_losses,
            'kl_divergences': self.kl_divergences,
            'epsilon': self.epsilon
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.network.load_state_dict(checkpoint['network'])


def main_optimized_ppo_training():
    """
    优化版PPO主训练函数
    
    主要优化：
    1. 智能奖励塑形
    2. 放松KL散度限制
    3. 合理的学习率和批量大小
    4. 改进的网络架构
    """
    print("=== 优化版PPO赛车轨道训练 ===")
    print(f"🎲 使用固定随机种子: {RANDOM_SEED}")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建优化版PPO智能体
    agent = OptimizedPPORacetrackAgent(
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,     # 标准裁剪
        ppo_epochs=4,       # 标准更新轮数
        batch_size=128,     # 增大批量
        buffer_size=1024,   # 增大缓冲区
        hidden_dim=128      # 适中网络
    )
    
    print(f"优化版PPO配置:")
    print(f"  - 学习率: 3e-4 (标准)")
    print(f"  - PPO轮数: 4 (标准)")
    print(f"  - 批量大小: 128 (增大)")
    print(f"  - 缓冲区大小: 1024 (增大)")
    print(f"  - 隐藏层维度: 128 (适中)")
    print(f"  - 裁剪比例: 0.2 (标准)")
    print(f"  - 目标KL散度: 0.5 (放松)")
    print(f"  - 奖励塑形: 智能前进奖励")
    
    # 训练前基准测试
    print("\n=== 训练前基准 ===")
    reward_before, steps_before, _, success_before = agent.test_episode()
    print(f"基准性能: 奖励={reward_before:.1f}, 步数={steps_before}, 成功={success_before}")
    
    # 训练设置
    n_episodes = 2000
    
    print(f"\n=== 开始优化版PPO训练 ===")
    print(f"训练轮数: {n_episodes}")
    
    # 训练统计
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=50)
    
    # 最佳模型保护
    best_success_rate = 0.0
    best_model_state = None
    patience = 0
    
    for episode in range(n_episodes):
        # 训练一个episode
        reward, steps, success = agent.train_episode(episode)
        
        agent.episode_rewards.append(reward)
        agent.episode_steps.append(steps)
        success_window.append(1 if success else 0)
        reward_window.append(reward)
        
        current_success_rate = np.mean(success_window)
        agent.success_rate.append(current_success_rate)
        
        # 最佳模型保护
        if episode >= 50 and current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            best_model_state = {
                'network': agent.network.state_dict().copy(),
                'optimizer': agent.optimizer.state_dict().copy(),
                'episode': episode,
                'success_rate': current_success_rate
            }
            patience = 0
            print(f"💾 保存最佳模型: Episode {episode+1}, 成功率={current_success_rate:.3f}")
        else:
            patience += 1
        
        # 性能退化检测
        if patience > agent.max_patience and best_model_state:
            print(f"\n⚠️ 性能停滞，恢复最佳模型...")
            agent.network.load_state_dict(best_model_state['network'])
            agent.optimizer.load_state_dict(best_model_state['optimizer'])
            print(f"   已恢复Episode {best_model_state['episode']+1}的模型")
            patience = 0
            agent.epsilon = max(0.05, agent.epsilon * 1.2)  # 增加探索
        
        # 定期输出训练进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(agent.episode_steps[-25:])
            avg_policy_loss = np.mean(agent.policy_losses[-10:]) if agent.policy_losses else 0
            avg_value_loss = np.mean(agent.value_losses[-10:]) if agent.value_losses else 0
            avg_kl_div = np.mean(agent.kl_divergences[-10:]) if agent.kl_divergences else 0
            
            print(f"Episode {episode + 1:4d}: "
                  f"奖励={avg_reward:6.1f}, 步数={avg_steps:5.1f}, "
                  f"成功率={current_success_rate:.3f}, ε={agent.epsilon:.3f}")
            print(f"                     策略损失={avg_policy_loss:.4f}, "
                  f"价值损失={avg_value_loss:.4f}, KL散度={avg_kl_div:.4f}")
            print(f"                     最佳成功率={best_success_rate:.3f}, 耐心值={patience}")
    
    # 恢复最佳模型进行最终测试
    if best_model_state:
        print(f"\n🔄 恢复最佳模型进行最终测试...")
        agent.network.load_state_dict(best_model_state['network'])
    
    # 最终测试
    print(f"\n=== 最终评估 ===")
    test_results = []
    for i in range(50):
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
    
    final_success_rate = np.mean([r[2] for r in test_results])
    final_avg_reward = np.mean([r[0] for r in test_results])
    final_avg_steps = np.mean([r[1] for r in test_results])
    
    print(f"优化版PPO最终结果（50次测试）:")
    print(f"  成功率: {final_success_rate:.1%}")
    print(f"  平均奖励: {final_avg_reward:.1f}")
    print(f"  平均步数: {final_avg_steps:.1f}")
    
    # 与之前版本对比
    print(f"\n📊 对比结果:")
    print(f"  原版PPO成功率: 12%")
    print(f"  稳定PPO成功率: 8%")
    print(f"  优化PPO成功率: {final_success_rate:.1%}")
    print(f"  Actor-Critic成功率: 62%")
    print(f"  Sarsa(λ)成功率: 90%")
    
    if final_success_rate > 0.5:
        print("🎉 优化大成功！PPO性能显著提升")
    elif final_success_rate > 0.3:
        print("✅ 优化效果显著，PPO性能大幅改善")
    elif final_success_rate > 0.15:
        print("⚖️ 优化有效，但仍有改进空间")
    else:
        print("⚠️ 需要进一步优化")
    
    # 保存模型
    agent.save_model("models/optimized_ppo_racetrack_model.pth")
    print(f"优化版PPO模型已保存")
    
    # 展示一个成功的路径
    if final_success_rate > 0:
        print(f"\n=== 展示最优路径 ===")
        best_reward = -float('inf')
        best_path = None
        best_steps = 0
        
        for i in range(10):
            reward, steps, path, success = agent.test_episode()
            if success and reward > best_reward:
                best_reward = reward
                best_path = path
                best_steps = steps
        
        if best_path:
            print(f"最优路径: 奖励={best_reward:.1f}, 步数={best_steps}")
            print(f"路径长度: {len(best_path)}")
            print(f"起点: {best_path[0]}")
            print(f"终点: {best_path[-1]}")
            
            # 可视化路径
            agent.test_episode(render=True)
    
    return agent, test_results


if __name__ == "__main__":
    # 运行优化版PPO训练
    main_optimized_ppo_training() 