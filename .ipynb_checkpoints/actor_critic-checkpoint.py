"""
Actor-Critic 强化学习智能体 - 解决性能退化问题的优化版本

本文件实现了一个针对赛车轨道环境的Actor-Critic智能体，
主要解决了训练过程中的性能退化问题。

核心改进：
1. 分阶段训练策略
2. 最佳模型保护机制  
3. 极慢探索率衰减
4. 分离的Actor-Critic优化器
5. 性能监控与自动恢复

作者：AI Assistant
最后更新：2024年
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from collections import deque
from racetrack_env import RacetrackEnv

# 设置随机种子确保结果可重现
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
# 确保PyTorch的确定性行为
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"🎲 随机种子已设置为: {RANDOM_SEED}")


class SharedNetwork(nn.Module):
    """
    Actor-Critic共享网络架构
    
    采用共享底层特征提取 + 分离头部的设计：
    - 共享层：提取环境状态的通用特征
    - Actor头部：输出动作概率分布
    - Critic头部：估计状态价值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SharedNetwork, self).__init__()
        
        # 共享的底层特征提取网络
        # 使用两层全连接网络，逐渐压缩特征维度
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 压缩到一半维度
            nn.ReLU()
        )
        
        # Actor头部：输出动作logits
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic头部：输出状态价值估计
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim] 或 [state_dim]
            
        Returns:
            action_probs: 动作概率分布 [batch_size, action_dim] 或 [action_dim]
            value: 状态价值估计 [batch_size, 1] 或 [1]
        """
        # 提取共享特征
        shared_features = self.shared_layers(state)
        
        # 计算动作logits并转换为概率分布
        action_logits = self.actor_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 计算状态价值
        value = self.critic_head(shared_features)
        
        return action_probs, value


class Experience:
    """
    经验回放中的单个经验样本
    
    存储一个完整的状态转移四元组：(s, a, r, s', done)
    外加用于策略梯度的log_prob
    """
    def __init__(self, state, action, reward, next_state, done, log_prob):
        self.state = state          # 当前状态
        self.action = action        # 执行的动作
        self.reward = reward        # 获得的奖励
        self.next_state = next_state # 下一个状态
        self.done = done            # 是否终止
        self.log_prob = log_prob    # 动作的对数概率


class OptimizedActorCriticAgent:
    """
    优化的Actor-Critic智能体
    
    主要特性：
    1. 解决性能退化问题：通过极慢探索衰减和最佳模型保护
    2. 稳定的价值函数学习：分离优化器，降低Critic学习率
    3. 改进的状态表示：包含目标方向和速度对齐信息
    4. 严格的动作掩码：完全禁止碰撞动作
    5. 简化的奖励塑形：避免过度工程化
    """
    
    def __init__(self, env: RacetrackEnv, lr=0.001, gamma=0.99, 
                 hidden_dim=128, buffer_size=128, gae_lambda=0.95):
        """
        初始化智能体
        
        Args:
            env: 赛车轨道环境
            lr: 学习率（主要用于兼容性，实际使用分离的优化器）
            gamma: 折扣因子
            hidden_dim: 隐藏层维度
            buffer_size: 经验缓冲区大小
            gae_lambda: GAE的λ参数
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        
        # 状态特征维度：8维精心设计的特征
        # [norm_x, norm_y, norm_vx, norm_vy, norm_distance, 
        #  goal_direction_x, goal_direction_y, velocity_alignment]
        self.state_dim = 8
        self.action_dim = env.n_actions
        
        # 创建共享网络
        self.network = SharedNetwork(self.state_dim, self.action_dim, hidden_dim)
        
        # 经验缓冲区：使用双端队列实现固定大小缓冲
        self.buffer = deque(maxlen=buffer_size)
        
        # 探索策略参数
        self.epsilon = 0.5              # 初始探索率（适中）
        self.epsilon_min = 0.15         # 最小探索率（保持足够探索）
        self.epsilon_decay = 0.9998     # 极慢衰减（防止过早收敛）
        
        # 训练稳定性参数
        self.entropy_coef = 0.05        # 熵正则化系数（防止过拟合）
        self.update_frequency = 32      # 批量更新频率
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.losses: List[float] = []
        self.value_losses: List[float] = []
        self.policy_losses: List[float] = []
        
        # 关键改进：分离的Actor-Critic优化器
        # Actor使用较高学习率（策略学习）
        self.actor_optimizer = optim.AdamW(
            self.network.actor_head.parameters(), 
            lr=0.0005, 
            weight_decay=1e-5
        )
        # Critic使用较低学习率（防止价值函数过拟合）
        self.critic_optimizer = optim.AdamW(
            self.network.critic_head.parameters(), 
            lr=0.0003,
            weight_decay=1e-5
        )
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        将环境状态转换为神经网络输入张量
        
        设计8维特征向量，包含：
        1. 位置信息（归一化）
        2. 速度信息（归一化） 
        3. 目标距离（归一化）
        4. 目标方向（单位向量）
        5. 速度与目标方向的对齐度
        
        Args:
            state: 环境状态 (x, y, vx, vy)
            
        Returns:
            torch.Tensor: 8维特征向量
        """
        x, y, vx, vy = state
        
        # 1. 基础特征归一化到[0,1]范围
        norm_x = x / 31.0               # x坐标归一化
        norm_y = y / 16.0               # y坐标归一化
        norm_vx = vx / self.env.max_speed  # x方向速度归一化
        norm_vy = vy / self.env.max_speed  # y方向速度归一化
        
        # 2. 计算到最近终点的距离和方向
        min_distance = float('inf')
        goal_direction_x, goal_direction_y = 0, 0
        
        # 遍历所有终点，找到最近的
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            if distance < min_distance:
                min_distance = distance
                if distance > 0:
                    # 计算指向目标的单位方向向量
                    # 注意：使用环境的坐标系统
                    goal_direction_x = -(goal_x - x) / distance  # 向上为正
                    goal_direction_y = (goal_y - y) / distance   # 向右为正
        
        # 3. 距离归一化（使用对角线距离作为最大值）
        max_distance = np.sqrt(31**2 + 16**2)
        norm_distance = min_distance / max_distance
        
        # 4. 计算速度与目标方向的对齐度
        velocity_alignment = 0.0
        if min_distance > 0:
            velocity_mag = np.sqrt(vx**2 + vy**2)
            if velocity_mag > 0:
                # 当前速度的单位方向向量
                vel_dir_x = vx / velocity_mag
                vel_dir_y = vy / velocity_mag
                # 计算对齐度（点积，范围[-1,1]，取非负部分）
                velocity_alignment = max(0, vel_dir_x * goal_direction_x + vel_dir_y * goal_direction_y)
        
        # 返回8维特征向量
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_distance, goal_direction_x, goal_direction_y, 
            velocity_alignment
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], training=True) -> Tuple[int, torch.Tensor]:
        """
        选择动作（支持训练和测试模式）
        
        训练模式：使用ε-贪心策略，在有效动作中探索
        测试模式：纯贪心策略，选择概率最高的动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            action: 选择的动作索引
            log_prob: 动作的对数概率（用于策略梯度）
        """
        # 设置网络为评估模式（关闭dropout等）
        self.network.eval()
        
        # 转换状态为张量
        state_tensor = self.state_to_tensor(state)
        
        # 前向传播获取动作概率
        if training:
            action_probs, _ = self.network(state_tensor)
        else:
            with torch.no_grad():  # 测试时不需要梯度
                action_probs, _ = self.network(state_tensor)
        
        # 应用动作掩码（禁止碰撞动作）
        action_probs = self._apply_strict_action_mask(state, action_probs)
        
        # 动作选择策略
        if training and random.random() < self.epsilon:
            # 训练模式 + 随机探索：在有效动作中随机选择
            valid_actions = (action_probs > 0).nonzero().squeeze(-1)
            if len(valid_actions) > 0:
                action = valid_actions[random.randint(0, len(valid_actions)-1)]
            else:
                action = torch.argmax(action_probs)
        else:
            # 贪心策略：选择概率最高的动作
            action = torch.argmax(action_probs)
        
        # 计算动作的对数概率（用于策略梯度）
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def _apply_strict_action_mask(self, state: Tuple[int, int, int, int], 
                                action_probs: torch.Tensor) -> torch.Tensor:
        """
        应用严格的动作掩码，完全禁止会导致碰撞的动作
        
        这是安全机制，确保智能体不会选择明显错误的动作
        
        Args:
            state: 当前状态
            action_probs: 原始动作概率分布
            
        Returns:
            torch.Tensor: 掩码后的动作概率分布
        """
        x, y, vx, vy = state
        mask = torch.ones_like(action_probs)
        
        # 遍历所有可能的动作
        for i, (ax, ay) in enumerate(self.env.actions):
            # 预测执行动作后的新速度
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            
            # 处理速度为0的特殊情况（环境规则）
            if new_vx == 0 and new_vy == 0 and (x, y) not in self.env.start_positions:
                new_vx = 1
                new_vy = 1
            
            # 预测下一步位置
            new_x = x - new_vx  # 向上移动（x减小）
            new_y = y + new_vy  # 向右移动（y增大）
            
            # 检查是否会发生碰撞
            if self.env._check_collision(x, y, new_x, new_y):
                mask[i] = 0.0  # 禁止此动作
        
        # 确保至少有一个动作可选（安全措施）
        if mask.sum() == 0:
            mask.fill_(1.0)
        
        # 重新归一化概率分布
        masked_probs = action_probs * mask
        return masked_probs / (masked_probs.sum() + 1e-8)
    
    def _improved_reward_shaping(self, state, next_state, reward, done, steps):
        """
        简化的奖励塑形
        
        设计原则：
        1. 保持简单，避免过度工程化
        2. 只对关键事件给予奖励/惩罚
        3. 轻微的进步奖励，避免引导偏差
        
        Args:
            state: 当前状态
            next_state: 下一状态
            reward: 原始奖励
            done: 是否终止
            steps: 当前步数
            
        Returns:
            float: 塑形后的奖励
        """
        bonus = 0.0
        
        # 1. 成功/失败的明确奖励
        if done and reward > 0:
            bonus += 100    # 成功到达终点
        elif reward == -10:  # 碰撞
            bonus -= 50     # 碰撞惩罚
        
        # 2. 简单的进步奖励（距离减少）
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        # 计算到最近目标的曼哈顿距离
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        # 只有显著进步才给奖励（避免噪声）
        if curr_dist - next_dist > 1:
            bonus += 2.0
        
        # 3. 轻微的步数惩罚（鼓励效率）
        bonus -= 0.1
        
        return reward + bonus
    
    def _batch_update(self):
        """
        批量更新网络参数
        
        使用Actor-Critic算法：
        1. 计算GAE优势估计
        2. 分别更新Critic（价值函数）和Actor（策略）
        3. 使用梯度裁剪防止梯度爆炸
        """
        if len(self.buffer) < self.update_frequency:
            return
        
        # 设置网络为训练模式
        self.network.train()
        
        # 准备批量数据
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        
        for exp in self.buffer:
            states.append(self.state_to_tensor(exp.state))
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(self.state_to_tensor(exp.next_state))
            dones.append(exp.done)
            log_probs.append(exp.log_prob)
        
        # 转换为张量
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        
        # 重新计算当前策略下的价值和概率
        action_probs_batch, values = self.network(states)
        _, next_values = self.network(next_states)
        
        values = values.squeeze()
        next_values = next_values.squeeze()
        
        # 计算GAE优势
        advantages = self._compute_gae_fixed(rewards, values, next_values, dones)
        
        # 优势标准化（提高稳定性）
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # 裁剪优势值，防止过大的策略更新
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # 计算价值目标（TD目标）
        clipped_rewards = torch.clamp(rewards, -20, 20)  # 奖励裁剪
        
        td_targets = torch.zeros_like(clipped_rewards)
        for t in range(len(clipped_rewards)):
            if t == len(clipped_rewards) - 1:
                next_value = 0 if dones[t] else next_values[t].detach()
            else:
                next_value = values[t + 1].detach() * (1 - dones[t])
            td_targets[t] = clipped_rewards[t] + self.gamma * next_value
        
        value_targets = td_targets
        
        # 1. 更新Critic（价值函数）
        critic_loss = F.mse_loss(values, value_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.network.critic_head.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 2. 更新Actor（策略）
        # 重新计算动作概率（当前策略）
        action_dist = torch.distributions.Categorical(action_probs_batch)
        new_log_probs = action_dist.log_prob(actions)
        
        # 策略损失（策略梯度）
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        
        # 熵正则化（鼓励探索）
        entropy = action_dist.entropy().mean()
        actor_total_loss = actor_loss - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.actor_head.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 记录损失（用于监控）
        total_loss = actor_total_loss + 0.3 * critic_loss
        self.losses.append(total_loss.item())
        self.value_losses.append(critic_loss.item())
        self.policy_losses.append(actor_loss.item())
        
        # 部分清空缓冲区，保留一些经验
        for _ in range(self.update_frequency // 2):
            if len(self.buffer) > 0:
                self.buffer.popleft()
    
    def _compute_gae_fixed(self, rewards, values, next_values, dones):
        """
        计算广义优势估计（GAE）
        
        GAE结合了TD误差和蒙特卡洛估计，平衡了偏差和方差
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            next_values: 下一状态价值估计序列
            dones: 终止标志序列
            
        Returns:
            torch.Tensor: GAE优势估计
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else next_values[t]
            else:
                next_value = values[t + 1] * (1 - dones[t])
            
            # TD误差
            delta = rewards[t] + self.gamma * next_value - values[t]
            # GAE递推公式
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """
        训练单个episode
        
        Args:
            episode_num: episode编号
            
        Returns:
            total_reward: 总奖励
            steps: 步数
            success: 是否成功
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 200  # 最大步数限制
        
        episode_buffer = []
        last_reward = 0
        
        while steps < max_steps:
            # 选择动作
            action, log_prob = self.select_action(state, training=True)
            prev_state = state
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            last_reward = reward
            
            # 奖励塑形
            shaped_reward = self._improved_reward_shaping(prev_state, next_state, reward, done, steps)
            
            # 存储经验
            exp = Experience(prev_state, action, shaped_reward, next_state, done, log_prob)
            episode_buffer.append(exp)
            
            total_reward += reward  # 使用原始奖励计算回报
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 将episode经验添加到缓冲区
        self.buffer.extend(episode_buffer)
        
        # 批量更新
        if len(self.buffer) >= self.update_frequency:
            self._batch_update()
        
        # 更新探索率（每10个episode更新一次）
        if episode_num % 10 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 判断成功（到达终点且在步数限制内）
        success = (steps < max_steps and done and last_reward == 100)
        return total_reward, steps, success
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """
        测试单个episode（不训练）
        
        Args:
            render: 是否渲染轨迹
            
        Returns:
            total_reward: 总奖励
            steps: 步数
            path: 轨迹路径
            success: 是否成功
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]  # 记录位置轨迹
        max_steps = 300
        
        last_reward = 0
        with torch.no_grad():  # 测试时不需要梯度
            while steps < max_steps:
                action, _ = self.select_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                last_reward = reward
                
                if done:
                    break
                
                state = next_state
        
        success = (steps < max_steps and done and last_reward == 100)
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success
    
    def save_model(self, filepath: str):
        """保存模型和训练统计"""
        save_dict = {
            'network': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rate': self.success_rate,
            'losses': self.losses,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses,
            'epsilon': self.epsilon
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()


def main_fixed_degradation():
    """
    解决性能退化问题的主训练函数
    
    核心策略：
    1. 分阶段训练：高探索 → 平衡 → 精调
    2. 最佳模型保护：自动保存并恢复历史最佳性能
    3. 性能监控：实时检测退化并采取措施
    4. 分离优化器：Actor和Critic使用不同的学习率
    5. 随机种子控制，确保可重现性
    """
    print("=== 解决性能退化问题的训练（随机种子版本）===")
    print(f"🎲 使用固定随机种子: {RANDOM_SEED}")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建智能体
    agent = OptimizedActorCriticAgent(
        env=env,
        lr=0.0003,         # 基础学习率
        gamma=0.99,        # 折扣因子
        hidden_dim=128,    # 网络隐藏层维度
        buffer_size=512,   # 经验缓冲区大小
        gae_lambda=0.95    # GAE参数
    )
    
    # 重新设置探索策略（关键改进）
    agent.epsilon = 0.5                # 适中的初始探索率
    agent.epsilon_decay = 0.9998       # 极慢的衰减
    agent.epsilon_min = 0.15           # 保持较高的最小探索率
    agent.entropy_coef = 0.05          # 增加熵正则化
    
    # 性能保护机制
    best_success_rate = 0.0
    best_model_state = None
    patience = 0
    max_patience = 200
    performance_window = deque(maxlen=50)
    
    print(f"关键改进:")
    print(f"  - 极慢探索衰减(0.9998)，保持长期探索")
    print(f"  - 更高最小探索率(0.15)，防止过早收敛") 
    print(f"  - 增大缓冲区(512)，保留更多经验")
    print(f"  - 增强熵正则化(0.05)，防止过拟合")
    print(f"  - 添加早停机制，防止性能退化")
    
    # 训练前基准测试
    print("\n=== 训练前基准 ===")
    reward_before, steps_before, _, success_before = agent.test_episode()
    print(f"基准性能: 奖励={reward_before:.1f}, 步数={steps_before}, 成功={success_before}")
    
    # 分阶段训练设置
    print(f"\n=== 开始改进训练 ===")
    n_episodes = 2500
    stage1_episodes = 800   # 阶段1：高探索率训练
    stage2_episodes = 1200  # 阶段2：平衡训练  
    stage3_episodes = 500   # 阶段3：精调训练
    
    print(f"分阶段训练计划:")
    print(f"  阶段1 (0-{stage1_episodes}): 高探索率学习基础策略")
    print(f"  阶段2 ({stage1_episodes}-{stage1_episodes+stage2_episodes}): 平衡探索与利用")
    print(f"  阶段3 ({stage1_episodes+stage2_episodes}-{n_episodes}): 精调优化")
    
    # 训练循环
    all_rewards = []
    all_steps = []
    all_success_rates = []
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=50)
    
    for episode in range(n_episodes):
        # 分阶段调整参数
        if episode == stage1_episodes:
            print(f"\n🔄 进入阶段2: 降低Actor学习率，增强稳定性")
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] *= 0.7
                
        elif episode == stage1_episodes + stage2_episodes:
            print(f"\n🔧 进入阶段3: 进入精调模式")
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] *= 0.5
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] *= 0.7
        
        # 训练一个episode
        reward, steps, success = agent.train_episode(episode)
        all_rewards.append(reward)
        all_steps.append(steps)
        
        success_window.append(1 if success else 0)
        reward_window.append(reward)
        current_success_rate = np.mean(success_window)
        all_success_rates.append(current_success_rate)
        
        # 性能保护机制
        if episode >= 100:
            performance_window.append(current_success_rate)
            
            # 发现新的最佳性能时保存模型
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                best_model_state = {
                    'network': agent.network.state_dict().copy(),
                    'actor_optimizer': agent.actor_optimizer.state_dict().copy(),
                    'critic_optimizer': agent.critic_optimizer.state_dict().copy(),
                    'episode': episode,
                    'success_rate': current_success_rate
                }
                patience = 0
                print(f"💾 保存最佳模型: Episode {episode+1}, 成功率={current_success_rate:.3f}")
            else:
                patience += 1
            
            # 检测性能退化并恢复
            if len(performance_window) == 50:
                recent_performance = np.mean(list(performance_window)[-25:])
                early_performance = np.mean(list(performance_window)[:25])
                
                if recent_performance < early_performance * 0.7 and patience > max_patience:
                    print(f"\n⚠️ 检测到性能退化！")
                    print(f"   早期性能: {early_performance:.3f}")
                    print(f"   最近性能: {recent_performance:.3f}")
                    print(f"   恢复最佳模型...")
                    
                    if best_model_state:
                        agent.network.load_state_dict(best_model_state['network'])
                        agent.actor_optimizer.load_state_dict(best_model_state['actor_optimizer'])
                        agent.critic_optimizer.load_state_dict(best_model_state['critic_optimizer'])
                        print(f"   已恢复Episode {best_model_state['episode']+1}的模型")
                        
                        # 重置探索率，给予第二次机会
                        agent.epsilon = max(0.3, agent.epsilon * 1.5)
                        print(f"   重置探索率至{agent.epsilon:.3f}")
                        patience = 0
        
        # 定期输出训练进度
        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(all_steps[-25:])
            avg_loss = np.mean(agent.losses[-50:]) if agent.losses else 0
            
            stage_name = "阶段1" if episode < stage1_episodes else "阶段2" if episode < stage1_episodes + stage2_episodes else "阶段3"
            
            print(f"{stage_name} Episode {episode + 1:4d}: "
                  f"奖励={avg_reward:6.1f}, 步数={avg_steps:5.1f}, "
                  f"成功率={current_success_rate:.3f}, ε={agent.epsilon:.3f}")
            print(f"                     损失={avg_loss:.4f}, "
                  f"最佳成功率={best_success_rate:.3f}, 耐心={patience}")
            
            # 性能诊断
            if episode > 100:
                recent_window = list(success_window)[-25:]
                recent_success = np.mean(recent_window)
                if recent_success < 0.05:
                    print(f"🚨 最近25轮成功率仅{recent_success:.3f}，可能需要调整")
                elif recent_success > best_success_rate * 0.8:
                    print(f"✅ 表现良好，接近最佳水平")
    
    # 最终评估
    print(f"\n=== 训练完成，最终评估 ===")
    final_success = np.mean(all_success_rates[-100:]) if len(all_success_rates) >= 100 else 0
    print(f"最终100回合成功率: {final_success:.3f}")
    print(f"历史最佳成功率: {best_success_rate:.3f}")
    
    # 如果最终性能不如历史最佳，恢复最佳模型
    if best_model_state and final_success < best_success_rate * 0.8:
        print(f"\n🔄 最终性能不如历史最佳，恢复最佳模型进行测试...")
        agent.network.load_state_dict(best_model_state['network'])
        
    # 最终测试
    test_results = []
    for i in range(50):
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
    
    final_test_success = np.mean([r[2] for r in test_results])
    print(f"严格测试成功率（50次）: {final_test_success:.3f}")
    
    # 评估结果
    if final_test_success > 0.6:
        print("🎉 性能退化问题解决！成功率超过60%")
    elif final_test_success > 0.4:
        print("✅ 性能明显改善，成功率超过40%") 
    elif final_test_success > 0.2:
        print("⚖️ 性能有所改善，但仍需进一步优化")
    else:
        print("⚠️ 问题仍然存在，需要更深层的架构改进")
    
    # 保存最终模型
    agent.save_model("models/fixed_degradation_model.pth")
    print(f"改进后模型已保存到 models/ 文件夹")
    
    return agent, test_results, all_success_rates


def main_advanced_tuning():
    """
    针对已达到60%成功率的模型进行进一步精调
    
    策略：
    1. 加载最佳模型作为起点
    2. 使用更细致的学习率调度
    3. 专注于最后10-20%的性能提升
    """
    print("=== 高级精调训练（基于60%成功率模型）===")
    
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    agent = OptimizedActorCriticAgent(env=env)
    
    # 尝试加载之前的最佳模型
    try:
        agent.load_model("models/fixed_degradation_model.pth")
        print("✅ 成功加载之前的训练模型")
    except:
        print("⚠️ 未找到之前的模型，从头开始训练")
    
    # 精调参数设置
    agent.epsilon = 0.1  # 低探索率，主要利用已学到的策略
    agent.epsilon_decay = 0.9995
    agent.epsilon_min = 0.05
    
    # 降低学习率进行精调
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = 0.0001
    for param_group in agent.critic_optimizer.param_groups:
        param_group['lr'] = 0.00005
    
    print(f"精调参数: 探索率={agent.epsilon}, Actor lr=0.0001, Critic lr=0.00005")
    
    # 精调训练
    best_test_success = 0.0
    for episode in range(500):  # 短期精调
        reward, steps, success = agent.train_episode(episode)
        
        # 每50个episode测试一次
        if (episode + 1) % 50 == 0:
            test_results = []
            for _ in range(20):
                _, _, _, test_success = agent.test_episode()
                test_results.append(test_success)
            
            current_success = np.mean(test_results)
            print(f"精调 Episode {episode+1}: 测试成功率={current_success:.3f}")
            
            if current_success > best_test_success:
                best_test_success = current_success
                agent.save_model("models/advanced_tuned_model.pth")
                print(f"💾 保存改进模型，成功率提升至{current_success:.3f}")
    
    return agent


if __name__ == "__main__":
    # 运行主训练函数
    agent, test_results, success_rates = main_fixed_degradation()
    
    # 如果性能还不错，可以尝试进一步精调
    final_success = np.mean([r[2] for r in test_results])
    if final_success >= 0.5:
        print(f"\n🚀 当前成功率{final_success:.1%}不错，尝试进一步精调...")
        main_advanced_tuning()