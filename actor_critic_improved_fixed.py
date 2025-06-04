import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from collections import deque
from racetrack_env import RacetrackEnv


class SharedNetwork(nn.Module):
    """共享底层的网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SharedNetwork, self).__init__()
        # 共享的底层特征提取
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor头部
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic头部  
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
        # 移除Dropout，避免训练时的随机扰动
        
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        return F.softmax(action_logits, dim=-1), value


class Experience:
    """经验缓冲区中的单个经验"""
    def __init__(self, state, action, reward, next_state, done, log_prob):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob


class OptimizedActorCriticAgent:
    """
    优化后的Actor-Critic智能体
    主要改进：
    1. 共享网络架构
    2. 经验回放缓冲区
    3. 改进的状态表示
    4. 更好的探索策略
    5. 优势标准化
    6. 修复碰撞处理
    """
    
    def __init__(self, env: RacetrackEnv, lr=0.003, gamma=0.99, 
                 hidden_dim=128, buffer_size=64, gae_lambda=0.95):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        
        # 修正状态特征维度
        self.state_dim = 8  # 简化但保留关键特征
        self.action_dim = env.n_actions
        
        # 创建共享网络
        self.network = SharedNetwork(self.state_dim, self.action_dim, hidden_dim)
        
        # 单一优化器
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度器（更激进的衰减）
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # 经验缓冲区
        self.buffer = deque(maxlen=buffer_size)
        
        # ε-贪心探索（替代温度探索）
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.losses: List[float] = []
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """改进的状态表示 - 解决坐标系统一问题"""
        x, y, vx, vy = state
        
        # 统一坐标系：使用环境的实际移动方式
        # vx > 0: 向上移动 (x 减小)
        # vy > 0: 向右移动 (y 增大)
        
        # 基础特征归一化（统一到[0,1]范围）
        norm_x = x / 31.0  # 0-31范围
        norm_y = y / 16.0  # 0-16范围
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
                    # 使用与环境一致的坐标系
                    goal_direction_x = -(goal_x - x) / distance  # 向上为正
                    goal_direction_y = (goal_y - y) / distance   # 向右为正
        
        # 统一距离归一化（使用实际最大距离）
        max_distance = np.sqrt(31**2 + 16**2)  # 对角线距离
        norm_distance = min_distance / max_distance
        
        # 速度与目标方向的对齐度
        velocity_alignment = 0.0
        if min_distance > 0:
            velocity_mag = np.sqrt(vx**2 + vy**2)
            if velocity_mag > 0:
                # 使用环境的移动定义：vx向上(负x)，vy向右(正y)
                vel_dir_x = vx / velocity_mag
                vel_dir_y = vy / velocity_mag
                velocity_alignment = max(0, vel_dir_x * goal_direction_x + vel_dir_y * goal_direction_y)
        
        # 是否接近终点
        near_goal = 1.0 if min_distance <= 3 else 0.0
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_distance, goal_direction_x, goal_direction_y, 
            velocity_alignment
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], training=True) -> Tuple[int, torch.Tensor]:
        """使用ε-贪心策略选择动作"""
        self.network.eval()  # 评估模式进行采样

        state_tensor = self.state_to_tensor(state)

        if training:
            # 训练时需要梯度，因此不使用 no_grad
            action_probs, _ = self.network(state_tensor)
        else:
            # 测试或评估时不需要梯度
            with torch.no_grad():
                action_probs, _ = self.network(state_tensor)

        # 应用严格的动作掩码
        action_probs = self._apply_strict_action_mask(state, action_probs)

        if training and random.random() < self.epsilon:
            # ε-贪心探索
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        else:
            # 贪心选择
            action = torch.argmax(action_probs)

        # 重新计算log_prob用于训练
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob
    
    def _apply_strict_action_mask(self, state: Tuple[int, int, int, int], 
                                action_probs: torch.Tensor) -> torch.Tensor:
        """严格的动作掩码 - 完全禁止必撞墙动作"""
        x, y, vx, vy = state
        mask = torch.ones_like(action_probs)
        
        for i, (ax, ay) in enumerate(self.env.actions):
            # 预测下一步位置
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            new_x = x - new_vx  # 向上移动
            new_y = y + new_vy  # 向右移动
            
            # 检查是否会碰撞
            if self.env._check_collision(x, y, new_x, new_y):
                mask[i] = 0.0  # 完全禁止
        
        # 确保至少有一个动作可选
        if mask.sum() == 0:
            mask.fill_(1.0)
        
        # 重新归一化
        masked_probs = action_probs * mask
        return masked_probs / (masked_probs.sum() + 1e-8)
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 300  # 减少最大步数，避免无意义的长轨迹
        
        episode_buffer = []
        
        last_reward = 0  # Track the last environment reward to determine success
        while steps < max_steps:
            # 选择动作
            action, log_prob = self.select_action(state, training=True)
            
            # 执行动作前记录当前状态
            prev_state = state
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            last_reward = reward  # remember raw reward before shaping
            
            # 修复：检测碰撞的正确方法
            # 如果奖励是-10且没有成功到达终点，说明发生了碰撞
            crashed = (reward == -10 and not done)
            if crashed:
                # 环境已经将智能体重置到起点，此处不再终止episode，
                # 允许智能体继续尝试以便从失败中学习
                pass
            
            # 改进的奖励塑形
            shaped_reward = self._improved_reward_shaping(prev_state, next_state, reward, done, steps)
            
            # 存储经验
            exp = Experience(prev_state, action, shaped_reward, next_state, done, log_prob)
            episode_buffer.append(exp)
            
            total_reward += reward  # 统计原始奖励
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 将episode经验加入缓冲区
        self.buffer.extend(episode_buffer)
        
        # 批量更新
        if len(self.buffer) >= self.buffer_size:
            self._batch_update()
            self.buffer.clear()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Determine success based on the final environment reward indicating
        # the goal was reached (reward == 100 from RacetrackEnv.step)
        success = (steps < max_steps and done and last_reward == 100)
        return total_reward, steps, success
    
    def _improved_reward_shaping(self, state, next_state, reward, done, steps):
        """改进的奖励塑形"""
        if done and reward > 0:
            return reward + 50  # 增加成功奖励
        
        # 如果reward为-10且未结束，说明发生了碰撞并被重置到起点
        if (reward == -10 and not done) or (done and reward < 0):
            return -20  # 碰撞惩罚
        
        # 进步奖励（加大权重）
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        progress_reward = (curr_dist - next_dist) * 3.0  # 进一步加大进步奖励
        
        # 速度奖励
        _, _, vx, vy = next_state
        speed_reward = min(vx + vy, 4) * 0.3
        
        # 大幅减少步数惩罚
        step_penalty = -0.02  # 从-0.05进一步减少到-0.02
        
        # 接近目标的指数奖励
        proximity_bonus = 0.0
        if next_dist <= 5:
            proximity_bonus = (6 - next_dist) * 2.0
        
        # 如果在终点附近，给予额外奖励
        if next_dist <= 2:
            proximity_bonus += 10.0
        
        shaped_reward = step_penalty + progress_reward + speed_reward + proximity_bonus
        
        return shaped_reward
    
    def _batch_update(self):
        """批量更新网络"""
        if len(self.buffer) == 0:
            return
        
        self.network.train()  # 训练模式
        
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
        
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        
        # 计算价值和下一步价值
        _, values = self.network(states)
        _, next_values = self.network(next_states)
        
        values = values.squeeze()
        next_values = next_values.squeeze()
        
        # 计算GAE优势
        advantages = self._compute_gae(rewards, values, next_values, dones)
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算损失
        value_targets = advantages + values.detach()
        
        critic_loss = F.mse_loss(values, value_targets)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        self.losses.append(total_loss.item())
    
    def _compute_gae(self, rewards, values, next_values, dones):
        """计算Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # Use the estimated value of the next state provided to this
            # function. When the current step is terminal, there is no
            # bootstrap value so we set it to 0.
            next_value = 0 if dones[t] else next_values[t]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int], List[float]]:
        """训练智能体"""
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        
        success_window = deque(maxlen=100)
        
        for episode in range(n_episodes):
            reward, steps, success = self.train_episode(episode)
            self.episode_rewards.append(reward)
            self.episode_steps.append(steps)
            
            success_window.append(1 if success else 0)
            current_success_rate = np.mean(success_window)
            self.success_rate.append(current_success_rate)
            
            # 学习率调度
            if episode % 500 == 0 and episode > 0:
                self.scheduler.step()
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_steps = np.mean(self.episode_steps[-50:])
                avg_loss = np.mean(self.losses[-50:]) if self.losses else 0
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, "
                      f"平均步数 = {avg_steps:.2f}, 成功率 = {current_success_rate:.3f}, "
                      f"损失 = {avg_loss:.4f}, ε = {self.epsilon:.3f}")
        
        return self.episode_rewards, self.episode_steps, self.success_rate
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """测试一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 300
        
        last_reward = 0  # Track the last reward to determine success
        with torch.no_grad():
            while steps < max_steps:
                action, _ = self.select_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                last_reward = reward  # store raw reward for success check
                
                if done:
                    break
                
                state = next_state
        
        # Successful episode if the final raw reward signals reaching the goal
        success = (steps < max_steps and done and last_reward == 100)
        
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
            'losses': self.losses,
            'epsilon': self.epsilon
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()


def main():
    """主函数"""
    print("=== 优化后的Actor-Critic算法演示 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 最大速度: {env.max_speed}")
    print(f"  - 动作数量: {env.n_actions}")
    
    # 创建优化的智能体
    print("\n=== 优化的Actor-Critic智能体 ===")
    agent = OptimizedActorCriticAgent(
        env=env,
        lr=0.003,          # 提高学习率
        gamma=0.99,
        hidden_dim=128,    # 减小网络大小
        buffer_size=64,    # 批量更新
        gae_lambda=0.95    # GAE
    )
    
    print(f"主要优化：")
    print(f"  - 共享网络架构，减少参数")
    print(f"  - 批量更新，减少方差")
    print(f"  - 严格动作掩码")
    print(f"  - ε-贪心探索策略")
    print(f"  - GAE优势估计")
    print(f"  - 修复碰撞处理")
    print(f"  - 改进奖励塑形")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before, success_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}, 成功 = {success_before}")
    
    # 训练智能体
    print(f"\n=== 开始训练优化版Actor-Critic ===")
    n_episodes = 1000  # 预期更快收敛
    rewards, steps, success_rates = agent.train(n_episodes=n_episodes, verbose=True)
    
    # 分析训练结果
    print(f"\n=== 训练结果分析 ===")
    print(f"总训练回合数: {n_episodes}")
    print(f"最终50回合平均奖励: {np.mean(rewards[-50:]):.2f}")
    print(f"最终50回合平均步数: {np.mean(steps[-50:]):.2f}")
    print(f"最终50回合成功率: {np.mean(success_rates[-50:]):.3f}")
    
    # 训练后测试
    print("\n=== 训练后测试 ===")
    test_results = []
    for i in range(10):
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
    
    avg_test_reward = np.mean([r[0] for r in test_results])
    avg_test_steps = np.mean([r[1] for r in test_results])
    test_success_rate = np.mean([r[2] for r in test_results])
    
    print(f"测试结果（10次平均）:")
    print(f"  - 平均奖励: {avg_test_reward:.2f}")
    print(f"  - 平均步数: {avg_test_steps:.2f}")
    print(f"  - 成功率: {test_success_rate:.3f}")
    
    # 性能对比
    print(f"\n=== 性能提升 ===")
    print(f"奖励提升: {avg_test_reward - reward_before:.2f}")
    print(f"步数变化: {avg_test_steps - steps_before:.0f}")
    print(f"成功率: {test_success_rate:.3f}")
    
    # 可视化测试
    print("\n=== 可视化测试 ===")
    print("运行可视化测试...")
    agent.test_episode(render=True)
    
    # 保存模型
    model_path = "optimized_actor_critic_model.pth"
    agent.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main() 