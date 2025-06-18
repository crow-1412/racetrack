"""
Q-Guided Actor-Critic算法

结合Q-Learning和Actor-Critic优势的创新方法

核心思想：
1. 第一阶段：用Q-Learning快速学习准确的Q值
2. 第二阶段：用Q表指导神经网络学习 
3. 第三阶段：神经网络独立优化策略

优势：
- 结合Q-Learning的精确性和Actor-Critic的泛化能力
- 三阶段训练策略实现知识迁移
- 动态权重调整确保平滑过渡
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Tuple, List, Dict, Any
from collections import deque
from racetrack_env import RacetrackEnv


class QGuidedNetwork(nn.Module):
    """Q-Guided网络：三个输出头的混合架构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 三个输出头
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)  # 策略
        self.critic_head = nn.Linear(hidden_dim // 2, 1)         # 状态价值
        self.q_head = nn.Linear(hidden_dim // 2, action_dim)     # Q值
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        value = self.critic_head(features)
        q_values = self.q_head(features)
        return action_probs, value, q_values


class QGuidedActorCritic:
    """Q-Guided Actor-Critic算法主类"""
    
    def __init__(self, env: RacetrackEnv, lr=0.002, gamma=0.98, 
                 alpha_q=0.3, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.alpha_q = alpha_q
        self.epsilon = epsilon
        
        # 网络和Q表
        self.network = QGuidedNetwork(10, env.n_actions)  # 增加到10维状态特征
        self.Q_table = {}  # Q-Learning表格
        
        # 优化器（提高学习率）
        self.actor_optimizer = optim.AdamW(self.network.actor_head.parameters(), lr=lr*1.0)
        self.critic_optimizer = optim.AdamW(self.network.critic_head.parameters(), lr=lr*0.8)
        self.q_optimizer = optim.AdamW(self.network.q_head.parameters(), lr=lr*1.5)
        
        # 训练阶段控制（调整比例，更快进入混合阶段）
        self.training_phase = "q_learning"
        self.phase_episodes = {"q_learning": 300, "hybrid": 400, "actor_critic": 200}
        self.current_episode = 0
        self.q_weight = 1.0    # Q表权重
        self.ac_weight = 0.0   # 神经网络权重
        
        # 经验缓冲（增大缓冲区）
        self.buffer = deque(maxlen=256)
        
        print("🚀 优化版Q-Guided Actor-Critic初始化完成")
        print(f"训练计划：Q-Learning({self.phase_episodes['q_learning']}) → "
              f"混合({self.phase_episodes['hybrid']}) → "
              f"Actor-Critic({self.phase_episodes['actor_critic']})")
    
    def state_to_tensor(self, state):
        """状态转特征向量（增强到10维）"""
        x, y, vx, vy = state
        
        # 基础归一化
        norm_x = x / 31.0
        norm_y = y / 16.0
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 到终点的距离和方向
        min_dist = float('inf')
        goal_dir_x, goal_dir_y = 0, 0
        best_goal = None
        
        for gx, gy in self.env.goal_positions:
            dist = np.sqrt((x-gx)**2 + (y-gy)**2)
            if dist < min_dist:
                min_dist = dist
                best_goal = (gx, gy)
                if dist > 0:
                    goal_dir_x = (gx-x) / dist  # 修正方向
                    goal_dir_y = (gy-y) / dist
        
        norm_dist = min_dist / np.sqrt(31**2 + 16**2)
        
        # 速度对齐度（朝向目标）
        vel_align = 0.0
        if min_dist > 0:
            vel_mag = np.sqrt(vx**2 + vy**2)
            if vel_mag > 0:
                vel_align = (vx*goal_dir_x + vy*goal_dir_y) / vel_mag
        
        # 新增特征：速度大小和到目标的曼哈顿距离
        vel_magnitude = np.sqrt(vx**2 + vy**2) / (self.env.max_speed * np.sqrt(2))
        manhattan_dist = (abs(x - best_goal[0]) + abs(y - best_goal[1])) / (31 + 16) if best_goal else 1.0
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_dist, goal_dir_x, goal_dir_y, vel_align,
            vel_magnitude, manhattan_dist  # 新增特征
        ], dtype=torch.float32)
    
    def get_q_value(self, state, action):
        """从Q表获取Q值"""
        key = (state, action)
        return self.Q_table.get(key, 0.0)
    
    def set_q_value(self, state, action, value):
        """设置Q表中的Q值"""
        self.Q_table[(state, action)] = value
    
    def update_phase(self):
        """更新训练阶段和权重"""
        if self.current_episode < self.phase_episodes["q_learning"]:
            self.training_phase = "q_learning"
            self.q_weight, self.ac_weight = 1.0, 0.0
        elif self.current_episode < sum(list(self.phase_episodes.values())[:2]):
            self.training_phase = "hybrid"
            # 更平滑的过渡，更早开始神经网络训练
            progress = (self.current_episode - self.phase_episodes["q_learning"]) / self.phase_episodes["hybrid"]
            self.q_weight = 1.0 - 0.8 * progress
            self.ac_weight = 0.8 * progress
        else:
            self.training_phase = "actor_critic"
            self.q_weight, self.ac_weight = 0.15, 1.0  # 保留更多Q表知识
    
    def select_action(self, state, training=True):
        """混合动作选择策略"""
        state_tensor = self.state_to_tensor(state)
        
        if training:
            if self.training_phase == "q_learning":
                # 纯Q-Learning策略
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.n_actions - 1)
                else:
                    q_vals = [self.get_q_value(state, a) for a in range(self.env.n_actions)]
                    action = int(np.argmax(q_vals))
                
                # 计算神经网络的log_prob（保持一致性）
                with torch.no_grad():
                    action_probs, _, _ = self.network(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(torch.tensor(action))
                    
            elif self.training_phase == "hybrid":
                # 混合策略：Q表 + 神经网络
                with torch.no_grad():
                    action_probs, _, nn_q_values = self.network(state_tensor)
                
                table_q_values = torch.tensor([self.get_q_value(state, a) for a in range(self.env.n_actions)])
                combined_q = self.q_weight * table_q_values + self.ac_weight * nn_q_values
                
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.n_actions - 1)
                else:
                    action = int(torch.argmax(combined_q))
                
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
                
            else:  # actor_critic阶段
                # 纯神经网络策略
                with torch.no_grad():
                    action_probs, _, _ = self.network(state_tensor)
                
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.n_actions - 1)
                else:
                    action = torch.argmax(action_probs).item()
                
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
        else:
            # 测试模式：优化策略，完全使用Q表（如果足够大）
            with torch.no_grad():
                action_probs, _, nn_q_values = self.network(state_tensor)
                
                if len(self.Q_table) > 200:  # 当Q表足够大时，完全依赖Q表
                    table_q_values = torch.tensor([self.get_q_value(state, a) for a in range(self.env.n_actions)])
                    # 如果Q表有足够知识，直接使用Q表，否则结合神经网络
                    if max(table_q_values) > 0:  # Q表有正值经验
                        action = int(torch.argmax(table_q_values))
                    else:
                        combined_q = 0.8 * table_q_values + 0.2 * nn_q_values
                        action = int(torch.argmax(combined_q))
                else:
                    # Q表不够大时，使用神经网络
                    action = torch.argmax(action_probs).item()
                
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
        
        return action, log_prob
    
    def q_learning_update(self, state, action, reward, next_state, done):
        """Q-Learning表格更新（优化版）"""
        current_q = self.get_q_value(state, action)
        
        if done:
            if reward == 100:  # 成功到达目标
                target_q = reward
            else:  # 其他终止情况
                target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.env.n_actions)]
            target_q = reward + self.gamma * max(next_q_values)
        
        # 动态学习率：成功经验使用更高的学习率
        dynamic_alpha = self.alpha_q * 1.5 if reward == 100 else self.alpha_q
        new_q = current_q + dynamic_alpha * (target_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def train_episode(self, episode_num):
        """训练单个episode"""
        self.current_episode = episode_num
        self.update_phase()
        
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 300
        
        while steps < max_steps:
            action, log_prob = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # Q-Learning更新（前两阶段）
            if self.training_phase in ["q_learning", "hybrid"]:
                self.q_learning_update(state, action, reward, next_state, done)
            
            # 存储经验
            self.buffer.append({
                'state': state,
                'action': action, 
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            if done:
                break
            state = next_state
        
        # 神经网络更新（后两阶段）
        if self.training_phase in ["hybrid", "actor_critic"] and len(self.buffer) >= 32:
            self.update_networks()
        
        # 探索率衰减（更快衰减）
        if episode_num % 8 == 0:  # 更频繁的衰减
            if self.training_phase == "q_learning":
                decay_rate = 0.995
            elif self.training_phase == "hybrid":
                decay_rate = 0.992  # 混合阶段更快衰减
            else:
                decay_rate = 0.990  # AC阶段最快衰减
            self.epsilon = max(0.01, self.epsilon * decay_rate)  # 更低的最小探索率
        
        success = (done and reward == 100)
        
        # 阶段转换提示
        if episode_num == self.phase_episodes["q_learning"]:
            print(f"🔄 进入混合训练阶段 (Episode {episode_num})")
            print(f"   Q表大小: {len(self.Q_table)}")
        elif episode_num == sum(list(self.phase_episodes.values())[:2]):
            print(f"🎭 进入Actor-Critic精调阶段 (Episode {episode_num})")
        
        return total_reward, steps, success
    
    def update_networks(self):
        """更新神经网络"""
        if len(self.buffer) < 32:
            return
        
        # 准备数据
        batch = list(self.buffer)[-32:]
        states = torch.stack([self.state_to_tensor(exp['state']) for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([self.state_to_tensor(exp['next_state']) for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        
        # 前向传播
        action_probs, values, q_values = self.network(states)
        _, next_values, _ = self.network(next_states)
        
        values = values.squeeze()
        next_values = next_values.squeeze()
        
        # 1. 更新Q网络（学习Q表知识）
        if self.training_phase == "hybrid" and len(self.Q_table) > 50:
            table_targets = []
            for exp in batch:
                table_q = self.get_q_value(exp['state'], exp['action'])
                table_targets.append(table_q)
            
            table_targets = torch.tensor(table_targets, dtype=torch.float32)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            q_loss = F.mse_loss(current_q, table_targets.detach())
            
            self.q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            self.q_optimizer.step()
        
        # 2. 更新Critic
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        critic_loss = F.mse_loss(values, td_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # 3. 更新Actor
        advantages = td_targets - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def test_episode(self, render=False):
        """测试episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        
        while steps < 300:
            action, _ = self.select_action(state, training=False)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            path.append(next_state[:2])
            
            if done:
                break
            state = next_state
        
        success = (done and reward == 100)
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success


class OptimizedQGuidedActorCritic(QGuidedActorCritic):
    """专门优化步数的Q-Guided Actor-Critic版本"""
    
    def __init__(self, env: RacetrackEnv, lr=0.003, gamma=0.99, 
                 alpha_q=0.4, epsilon=0.3):
        super().__init__(env, lr, gamma, alpha_q, epsilon)
        
        # 更激进的训练配置，重点在Q-Learning阶段
        self.phase_episodes = {"q_learning": 600, "hybrid": 200, "actor_critic": 100}
        
        print("🚀 专门优化步数的Q-Guided Actor-Critic")
        print(f"训练计划：Q-Learning({self.phase_episodes['q_learning']}) → "
              f"混合({self.phase_episodes['hybrid']}) → "
              f"Actor-Critic({self.phase_episodes['actor_critic']})")
    
    def get_step_bonus(self, steps_taken, max_steps=300):
        """基于步数的奖励修正"""
        # 步数越少，额外奖励越多
        step_efficiency = (max_steps - steps_taken) / max_steps
        return step_efficiency * 20  # 最多20分的效率奖励
    
    def train_episode(self, episode_num):
        """改进的训练episode，重点优化步数"""
        self.current_episode = episode_num
        self.update_phase()
        
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 300
        
        while steps < max_steps:
            action, log_prob = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            
            # 修正奖励：对短路径给予额外奖励
            if done and reward == 100:
                step_bonus = self.get_step_bonus(steps + 1)
                reward += step_bonus
            
            total_reward += reward
            steps += 1
            
            # Q-Learning更新（前两阶段）
            if self.training_phase in ["q_learning", "hybrid"]:
                self.q_learning_update(state, action, reward, next_state, done)
            
            # 存储经验
            self.buffer.append({
                'state': state,
                'action': action, 
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            if done:
                break
            state = next_state
        
        # 神经网络更新
        if self.training_phase in ["hybrid", "actor_critic"] and len(self.buffer) >= 64:
            self.update_networks()
        
        # 更快的探索率衰减
        if episode_num % 5 == 0:
            if self.training_phase == "q_learning":
                decay_rate = 0.993
            elif self.training_phase == "hybrid":
                decay_rate = 0.990
            else:
                decay_rate = 0.985
            self.epsilon = max(0.005, self.epsilon * decay_rate)
        
        success = (done and reward >= 100)  # 考虑奖励修正后的成功判断
        
        return total_reward, steps, success
    
    def select_action(self, state, training=True):
        """优化的动作选择，训练时更注重探索效率"""
        state_tensor = self.state_to_tensor(state)
        
        if training:
            if self.training_phase == "q_learning":
                # 改进的探索策略：基于Q值差异的探索
                q_vals = [self.get_q_value(state, a) for a in range(self.env.n_actions)]
                
                if len(q_vals) > 0 and max(q_vals) > min(q_vals):
                    # 如果有明显的Q值差异，降低探索率
                    effective_epsilon = self.epsilon * 0.5
                else:
                    # 如果Q值相近，保持探索
                    effective_epsilon = self.epsilon
                
                if random.random() < effective_epsilon:
                    action = random.randint(0, self.env.n_actions - 1)
                else:
                    action = int(np.argmax(q_vals))
                
                with torch.no_grad():
                    action_probs, _, _ = self.network(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(torch.tensor(action))
                    
            else:
                # 其他阶段使用原来的策略
                return super().select_action(state, training)
        else:
            # 测试模式：完全贪心，选择最优动作
            with torch.no_grad():
                action_probs, _, nn_q_values = self.network(state_tensor)
                
                if len(self.Q_table) > 100:
                    table_q_values = torch.tensor([self.get_q_value(state, a) for a in range(self.env.n_actions)])
                    # 如果Q表有经验，完全使用Q表
                    action = int(torch.argmax(table_q_values))
                else:
                    action = torch.argmax(action_probs).item()
                
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
        
        return action, log_prob


class StartPositionOptimizedQGAC(OptimizedQGuidedActorCritic):
    """针对起点位置优化的Q-Guided AC"""
    
    def __init__(self, env: RacetrackEnv, best_starts=None, lr=0.003, gamma=0.99, 
                 alpha_q=0.4, epsilon=0.3):
        super().__init__(env, lr, gamma, alpha_q, epsilon)
        
        # 设置最优起点（如果提供）
        self.best_starts = best_starts or [(31, 11), (31, 16), (31, 15)]  # 根据分析结果
        self.biased_training = True  # 偏向训练最优起点
        
        print("🎯 针对最优起点优化的Q-Guided AC")
        print(f"优先训练起点: {self.best_starts}")
    
    def reset_with_bias(self):
        """带偏向的重置，更多使用最优起点"""
        if self.biased_training and random.random() < 0.7:  # 70%概率使用最优起点
            start_pos = random.choice(self.best_starts)
            self.env.state = (start_pos[0], start_pos[1], 0, 0)
            return self.env.state
        else:
            return self.env.reset()  # 正常随机重置
    
    def train_episode(self, episode_num):
        """改进的训练，偏向最优起点"""
        self.current_episode = episode_num
        self.update_phase()
        
        # 使用偏向重置
        state = self.reset_with_bias()
        total_reward = 0.0
        steps = 0
        max_steps = 300
        
        while steps < max_steps:
            action, log_prob = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            
            # 加强奖励修正
            if done and reward == 100:
                step_bonus = self.get_step_bonus(steps + 1)
                # 对最优起点给予额外奖励
                if state[:2] in self.best_starts:
                    step_bonus *= 1.5  # 最优起点额外50%奖励
                reward += step_bonus
            
            total_reward += reward
            steps += 1
            
            # Q-Learning更新
            if self.training_phase in ["q_learning", "hybrid"]:
                self.q_learning_update(state, action, reward, next_state, done)
            
            # 存储经验
            self.buffer.append({
                'state': state,
                'action': action, 
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            if done:
                break
            state = next_state
        
        # 神经网络更新
        if self.training_phase in ["hybrid", "actor_critic"] and len(self.buffer) >= 64:
            self.update_networks()
        
        # 在训练后期减少起点偏向
        if episode_num > 400:
            self.biased_training = False
        
        # 探索率衰减
        if episode_num % 5 == 0:
            if self.training_phase == "q_learning":
                decay_rate = 0.993
            elif self.training_phase == "hybrid":
                decay_rate = 0.990
            else:
                decay_rate = 0.985
            self.epsilon = max(0.005, self.epsilon * decay_rate)
        
        success = (done and reward >= 100)
        return total_reward, steps, success


class UltraOptimizedQGAC(StartPositionOptimizedQGAC):
    """超级优化版Q-Guided AC - 激进优化步数"""
    
    def __init__(self, env: RacetrackEnv, best_starts=None, lr=0.004, gamma=0.995, 
                 alpha_q=0.5, epsilon=0.4):
        super().__init__(env, best_starts, lr, gamma, alpha_q, epsilon)
        
        # 重新初始化网络以支持15维特征
        self.network = QGuidedNetwork(15, env.n_actions)  # 升级到15维状态特征
        
        # 重新初始化优化器
        self.actor_optimizer = optim.AdamW(self.network.actor_head.parameters(), lr=lr*1.0)
        self.critic_optimizer = optim.AdamW(self.network.critic_head.parameters(), lr=lr*0.8)
        self.q_optimizer = optim.AdamW(self.network.q_head.parameters(), lr=lr*1.5)
        
        # 激进的训练配置 - 大幅延长Q-Learning阶段
        self.phase_episodes = {"q_learning": 1200, "hybrid": 150, "actor_critic": 50}
        
        # 多层级奖励系统
        self.step_bonus_multiplier = 3.0  # 更强的步数奖励
        self.efficiency_memory = deque(maxlen=100)  # 记录效率历史
        
        print("🚀 超级优化版Q-Guided Actor-Critic - 激进步数优化")
        print(f"训练计划：Q-Learning({self.phase_episodes['q_learning']}) → "
              f"混合({self.phase_episodes['hybrid']}) → "
              f"Actor-Critic({self.phase_episodes['actor_critic']})")
    
    def enhanced_step_bonus(self, steps_taken, max_steps=300):
        """增强的步数奖励系统"""
        # 基础效率奖励
        efficiency_ratio = (max_steps - steps_taken) / max_steps
        base_bonus = efficiency_ratio * 50 * self.step_bonus_multiplier
        
        # 历史效率对比奖励
        if self.efficiency_memory:
            avg_historical_steps = np.mean(self.efficiency_memory)
            if steps_taken < avg_historical_steps:
                improvement_bonus = (avg_historical_steps - steps_taken) * 2.0
                base_bonus += improvement_bonus
        
        # 极短路径的爆炸奖励
        if steps_taken <= 15:
            base_bonus += 100 * (16 - steps_taken)  # 15步内有爆炸奖励
        elif steps_taken <= 20:
            base_bonus += 50 * (21 - steps_taken)   # 20步内有巨额奖励
        elif steps_taken <= 25:
            base_bonus += 20 * (26 - steps_taken)   # 25步内有大额奖励
        
        return base_bonus
    
    def advanced_state_features(self, state):
        """高级状态特征工程 - 增强到15维"""
        x, y, vx, vy = state
        
        # 基础归一化特征
        norm_x = x / 31.0
        norm_y = y / 16.0
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 多目标距离和方向分析
        distances_to_goals = []
        directions_to_goals = []
        
        for gx, gy in self.env.goal_positions:
            dist = np.sqrt((x-gx)**2 + (y-gy)**2)
            distances_to_goals.append(dist)
            if dist > 0:
                directions_to_goals.append(((gx-x)/dist, (gy-y)/dist))
            else:
                directions_to_goals.append((0, 0))
        
        # 最近目标相关特征
        min_dist_idx = np.argmin(distances_to_goals)
        min_dist = distances_to_goals[min_dist_idx]
        best_dir_x, best_dir_y = directions_to_goals[min_dist_idx]
        
        # 速度对齐度（朝向最佳目标）
        vel_magnitude = np.sqrt(vx**2 + vy**2)
        vel_align = 0.0
        if vel_magnitude > 0 and min_dist > 0:
            vel_align = (vx*best_dir_x + vy*best_dir_y) / vel_magnitude
        
        # 路径规划特征
        norm_min_dist = min_dist / np.sqrt(31**2 + 16**2)
        manhattan_to_best = (abs(x - self.env.goal_positions[min_dist_idx][0]) + 
                           abs(y - self.env.goal_positions[min_dist_idx][1])) / (31 + 16)
        
        # 动态特征
        momentum_x = vx * norm_x  # 位置-速度交互
        momentum_y = vy * norm_y
        
        # 战术特征
        is_near_boundary = min(x, 31-x, y, 16-y) / 16.0  # 边界距离
        speed_efficiency = vel_magnitude / (self.env.max_speed * np.sqrt(2))
        
        # 新增：多步预测特征
        future_x = x - vx  # 预测下一步位置
        future_y = y + vy
        future_dist_to_goal = min([np.sqrt((future_x-gx)**2 + (future_y-gy)**2) 
                                 for gx, gy in self.env.goal_positions])
        future_dist_norm = future_dist_to_goal / np.sqrt(31**2 + 16**2)
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,           # 基础状态 (4)
            norm_min_dist, best_dir_x, best_dir_y,      # 目标相关 (3)
            vel_align, manhattan_to_best,               # 对齐和路径 (2)
            momentum_x, momentum_y,                     # 动态特征 (2)
            is_near_boundary, speed_efficiency,         # 战术特征 (2)
            future_dist_norm, vel_magnitude             # 预测特征 (2)
        ], dtype=torch.float32)  # 总共15维
    
    def state_to_tensor(self, state):
        """使用高级特征"""
        return self.advanced_state_features(state)
    
    def ultra_smart_action_selection(self, state, training=True):
        """超智能动作选择策略"""
        state_tensor = self.state_to_tensor(state)
        
        if training:
            if self.training_phase == "q_learning":
                # Q-Learning阶段：增强的智能探索
                q_vals = [self.get_q_value(state, a) for a in range(self.env.n_actions)]
                
                # 基于Q值方差的动态探索
                q_std = np.std(q_vals) if len(q_vals) > 1 else 0
                if q_std > 1.0:  # Q值差异明显时，更多利用
                    effective_epsilon = self.epsilon * 0.3
                else:  # Q值相近时，保持探索
                    effective_epsilon = self.epsilon
                
                # 距离导向的动作偏好
                if random.random() < effective_epsilon:
                    # 智能探索：偏向朝向目标的动作
                    x, y, vx, vy = state
                    goal_distances = []
                    for gx, gy in self.env.goal_positions:
                        goal_distances.append(np.sqrt((x-gx)**2 + (y-gy)**2))
                    
                    best_goal_idx = np.argmin(goal_distances)
                    gx, gy = self.env.goal_positions[best_goal_idx]
                    
                    # 计算朝向目标的理想速度变化
                    ideal_dvx = -1 if x > gx else (1 if x < gx else 0)
                    ideal_dvy = 1 if y < gy else (-1 if y > gy else 0)
                    
                    # 找到最接近理想方向的动作
                    best_action = 4  # 默认不变
                    best_score = -1
                    
                    for action_idx, (ax, ay) in enumerate(self.env.actions):
                        # 计算动作与理想方向的匹配度
                        score = ax * ideal_dvx + ay * ideal_dvy
                        if score > best_score:
                            best_score = score
                            best_action = action_idx
                    
                    action = best_action if random.random() < 0.7 else random.randint(0, self.env.n_actions - 1)
                else:
                    action = int(np.argmax(q_vals))
                
                # 计算log_prob
                with torch.no_grad():
                    action_probs, _, _ = self.network(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(torch.tensor(action))
            
            else:
                # 其他阶段使用原策略
                return super().select_action(state, training)
        
        else:
            # 测试模式：绝对贪心 + 智能回退
            with torch.no_grad():
                action_probs, _, nn_q_values = self.network(state_tensor)
                
                if len(self.Q_table) > 500:  # Q表足够大时
                    table_q_values = torch.tensor([self.get_q_value(state, a) for a in range(self.env.n_actions)])
                    
                    # 多策略融合
                    if max(table_q_values) > 5:  # Q表有高质量经验
                        action = int(torch.argmax(table_q_values))
                    else:
                        # 融合Q表和神经网络，偏向Q表
                        combined_q = 0.9 * table_q_values + 0.1 * nn_q_values
                        action = int(torch.argmax(combined_q))
                else:
                    # Q表较小时，使用神经网络
                    action = torch.argmax(action_probs).item()
                
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
        
        return action, log_prob
    
    def select_action(self, state, training=True):
        """重写动作选择"""
        return self.ultra_smart_action_selection(state, training)
    
    def train_episode(self, episode_num):
        """超级优化的训练episode"""
        self.current_episode = episode_num
        self.update_phase()
        
        # 使用偏向重置（前期更多最优起点）
        bias_probability = 0.9 if episode_num < 800 else 0.7
        if self.biased_training and random.random() < bias_probability:
            start_pos = random.choice(self.best_starts)
            self.env.state = (start_pos[0], start_pos[1], 0, 0)
            state = self.env.state
        else:
            state = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        max_steps = 300
        
        while steps < max_steps:
            action, log_prob = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            
            # 超级强化的奖励修正
            if done and reward == 100:
                step_bonus = self.enhanced_step_bonus(steps + 1)
                
                # 最优起点额外奖励
                if state[:2] in self.best_starts:
                    step_bonus *= 2.0  # 最优起点双倍奖励
                
                reward += step_bonus
                
                # 记录效率
                self.efficiency_memory.append(steps + 1)
            elif reward == -10:  # 碰撞
                reward -= 30  # 更严厉的碰撞惩罚
            
            total_reward += reward
            steps += 1
            
            # 强化Q-Learning更新
            if self.training_phase in ["q_learning", "hybrid"]:
                # 使用动态学习率
                if reward > 100:  # 成功且有奖励
                    dynamic_alpha = self.alpha_q * 2.0
                elif reward == 100:  # 普通成功
                    dynamic_alpha = self.alpha_q * 1.5  
                else:
                    dynamic_alpha = self.alpha_q
                
                # 保存原始alpha并临时修改
                original_alpha = self.alpha_q
                self.alpha_q = dynamic_alpha
                self.q_learning_update(state, action, reward, next_state, done)
                self.alpha_q = original_alpha
            
            # 存储经验
            self.buffer.append({
                'state': state,
                'action': action, 
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            if done:
                break
            state = next_state
        
        # 神经网络更新（更频繁）
        if self.training_phase in ["hybrid", "actor_critic"] and len(self.buffer) >= 32:
            self.update_networks()
        
        # 更积极的探索率衰减
        if episode_num % 3 == 0:  # 更频繁衰减
            if self.training_phase == "q_learning":
                decay_rate = 0.996 if episode_num < 600 else 0.992
            elif self.training_phase == "hybrid":
                decay_rate = 0.985
            else:
                decay_rate = 0.980
            self.epsilon = max(0.002, self.epsilon * decay_rate)  # 更低最小值
        
        # 动态调整起点偏向
        if episode_num > 900:
            self.biased_training = False
        
        success = (done and reward >= 100)
        return total_reward, steps, success


def ultra_optimization_demo():
    """演示超级优化版Q-Guided Actor-Critic"""
    print("🚀 超级优化版Q-Guided Actor-Critic演示")
    print("=" * 60)
    
    # 1. 快速分析最优起点（简化版）
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 使用之前分析的最优起点
    best_starts = [(31, 10), (31, 13), (31, 16), (31, 3), (31, 6)]
    
    print(f"🎯 使用预分析的最优起点: {best_starts}")
    
    # 2. 创建超级优化智能体
    ultra_agent = UltraOptimizedQGAC(env, best_starts=best_starts)
    
    # 3. 训练
    print(f"\n🚀 开始超级优化训练...")
    total_episodes = sum(ultra_agent.phase_episodes.values())
    
    training_results = []
    
    for episode in range(total_episodes):
        total_reward, steps, success = ultra_agent.train_episode(episode)
        
        if success:
            training_results.append(steps)
        
        # 进度报告
        if (episode + 1) % 200 == 0:
            recent_successes = [s for s in training_results[-50:]]
            if recent_successes:
                avg_recent = np.mean(recent_successes)
                min_recent = min(recent_successes)
                print(f"Episode {episode+1} ({ultra_agent.training_phase}): "
                      f"Q表={len(ultra_agent.Q_table)}, "
                      f"ε={ultra_agent.epsilon:.4f}, "
                      f"近50次成功平均={avg_recent:.1f}步, "
                      f"最佳={min_recent}步")
            else:
                print(f"Episode {episode+1} ({ultra_agent.training_phase}): "
                      f"Q表={len(ultra_agent.Q_table)}, "
                      f"ε={ultra_agent.epsilon:.4f}")
    
    # 4. 最终测试
    print(f"\n📊 最终性能测试:")
    
    # 最优起点测试
    best_start_results = []
    for best_start in best_starts[:3]:
        test_results = []
        for _ in range(20):
            ultra_agent.env.state = (best_start[0], best_start[1], 0, 0)
            reward, steps, _, success = ultra_agent.test_episode()
            if success:
                test_results.append(steps)
        
        if test_results:
            avg_steps = np.mean(test_results)
            min_steps = min(test_results)
            best_start_results.extend(test_results)
            print(f"起点{best_start}: 平均{avg_steps:.1f}步, 最佳{min_steps}步 ({len(test_results)}/20成功)")
    
    # 随机起点测试
    random_results = []
    for _ in range(50):
        reward, steps, _, success = ultra_agent.test_episode()
        if success:
            random_results.append(steps)
    
    # 总结
    print(f"\n🎉 超级优化结果:")
    if best_start_results:
        best_avg = np.mean(best_start_results)
        best_min = min(best_start_results) 
        print(f"✅ 最优起点平均: {best_avg:.1f}步 ± {np.std(best_start_results):.1f}")
        print(f"✅ 最优起点最佳: {best_min}步")
    
    if random_results:
        random_avg = np.mean(random_results)
        random_min = min(random_results)
        print(f"📊 随机起点平均: {random_avg:.1f}步 ± {np.std(random_results):.1f}")
        print(f"📊 随机起点最佳: {random_min}步")
    
    # 对比之前的结果
    if best_start_results and random_results:
        improvement_avg = 26.0 - random_avg  # 对比之前的26.0步
        improvement_best = 23.0 - best_avg   # 对比之前的23.0步
        print(f"\n📈 改进幅度:")
        print(f"随机起点改进: {improvement_avg:.1f}步")
        print(f"最优起点改进: {improvement_best:.1f}步")
        
        if best_min <= 12:
            print(f"🏆 实现了接近理论最优的{best_min}步！")
    
    return ultra_agent, best_start_results, random_results


# 修改demo函数
def demo():
    """演示全面优化的Q-Guided Actor-Critic算法"""
    print("选择演示模式:")
    print("1. 原版全面分析 + 标准优化")
    print("2. 超级优化版本（激进步数优化）")
    
    # 由于是自动运行，直接使用超级优化版本
    print("🚀 自动选择超级优化版本")
    return ultra_optimization_demo()


if __name__ == "__main__":
    demo() 