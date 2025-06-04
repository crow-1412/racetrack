import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from racetrack_env import RacetrackEnv


class PolicyNetwork(nn.Module):
    """改进的策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        # 增加网络容量和深度
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        action_logits = self.fc4(x)
        return F.softmax(action_logits, dim=-1)


class ValueNetwork(nn.Module):
    """改进的价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value


class ImprovedREINFORCEAgent:
    """
    改进的REINFORCE智能体，修复收敛问题
    """
    
    def __init__(self, env: RacetrackEnv, lr=0.003, gamma=0.99, hidden_dim=256, use_baseline=True):
        self.env = env
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # 扩展状态特征维度，增加更多有用信息
        self.state_dim = 8  # 位置(2) + 速度(2) + 到终点距离(1) + 相对位置(2) + 动作mask(1)
        self.action_dim = env.n_actions
        
        # 创建策略网络，使用更大的网络
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=500, gamma=0.8)
        
        # 如果使用基线，创建价值网络
        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_dim)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)
            self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=500, gamma=0.8)
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        
        # 经验回放缓冲区（存储最近的轨迹）
        self.trajectory_buffer = []
        self.buffer_size = 10
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """将状态转换为增强特征张量"""
        x, y, vx, vy = state
        
        # 基础特征归一化
        norm_x = (x - 16) / 16.0  # 居中归一化
        norm_y = (y - 8) / 8.0    # 居中归一化
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 计算到最近终点的距离
        min_distance = float('inf')
        goal_direction_x, goal_direction_y = 0, 0
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            if distance < min_distance:
                min_distance = distance
                goal_direction_x = (goal_x - x) / 32.0  # 归一化方向
                goal_direction_y = (goal_y - y) / 17.0
        
        norm_distance = min_distance / (32 + 17)  # 最大可能距离归一化
        
        # 是否在边界附近（安全特征）
        safety_factor = 1.0
        if x <= 2 or x >= 30 or y <= 1 or y >= 15:
            safety_factor = 0.0
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_distance, goal_direction_x, goal_direction_y, safety_factor
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], temperature=1.0) -> Tuple[int, torch.Tensor]:
        """根据策略选择动作，添加温度参数"""
        state_tensor = self.state_to_tensor(state)
        action_probs = self.policy_net(state_tensor)
        
        # 添加温度缩放
        if temperature != 1.0:
            action_logits = torch.log(action_probs + 1e-8) / temperature
            action_probs = F.softmax(action_logits, dim=-1)
        
        # 添加动作掩码，避免明显的坏动作
        action_probs = self._apply_action_mask(state, action_probs)
        
        # 使用概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def _apply_action_mask(self, state: Tuple[int, int, int, int], action_probs: torch.Tensor) -> torch.Tensor:
        """应用动作掩码，减少明显的错误动作"""
        x, y, vx, vy = state
        
        # 创建动作掩码（避免撞墙）
        mask = torch.ones_like(action_probs)
        
        for i, (ax, ay) in enumerate(self.env.actions):
            # 预测下一步位置
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            new_x = x - new_vx
            new_y = y + new_vy
            
            # 如果会撞墙，降低概率
            if (new_x < 0 or new_x >= self.env.track_size[0] or 
                new_y < 0 or new_y >= self.env.track_size[1] or
                (new_x < self.env.track_size[0] and new_y < self.env.track_size[1] and 
                 self.env.track[new_x, new_y] == 1)):
                mask[i] = 0.1  # 大幅降低概率而不是完全禁止
        
        # 重新归一化
        masked_probs = action_probs * mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
        
        return masked_probs
    
    def get_value(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """获取状态价值"""
        if self.use_baseline:
            state_tensor = self.state_to_tensor(state)
            return self.value_net(state_tensor)
        else:
            return torch.tensor(0.0)
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练一个episode，添加探索策略"""
        state = self.env.reset()
        
        # 存储轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        total_reward = 0.0
        steps = 0
        max_steps = 500  # 限制最大步数
        
        # 探索温度调度
        temperature = max(0.5, 1.0 - episode_num / 2000)
        
        # 收集完整轨迹
        while steps < max_steps:
            action, log_prob = self.select_action(state, temperature)
            value = self.get_value(state) if self.use_baseline else torch.tensor(0.0)
            
            next_state, reward, done = self.env.step(action)
            
            # 修改奖励塑形
            shaped_reward = self._shape_reward(state, next_state, reward, done)
            
            states.append(state)
            actions.append(action)
            rewards.append(shaped_reward)
            log_probs.append(log_prob)
            values.append(value)
            
            total_reward += reward  # 使用原始奖励统计
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        # 如果没有到达终点，给一个小的负奖励
        success = (steps < max_steps and done)
        if not success:
            rewards[-1] -= 5  # 惩罚未完成
        
        # 存储轨迹到缓冲区
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'success': success
        }
        
        # 更新缓冲区
        self.trajectory_buffer.append(trajectory)
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
        
        # 批量更新（缓冲区满时或者成功时）
        if len(self.trajectory_buffer) >= self.buffer_size or success:
            self._batch_update()
        
        return total_reward, steps, success
    
    def _shape_reward(self, state, next_state, reward, done):
        """奖励塑形"""
        if done and reward > 0:
            return reward  # 成功奖励不变
        
        # 距离奖励
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        # 当前距离最近终点
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        # 如果靠近终点，给予奖励
        progress_reward = (curr_dist - next_dist) * 0.1
        
        # 速度奖励（鼓励保持适当速度）
        _, _, vx, vy = next_state
        speed_reward = min(vx + vy, 3) * 0.02  # 鼓励但不过度
        
        return reward + progress_reward + speed_reward
    
    def _batch_update(self):
        """批量更新网络"""
        if not self.trajectory_buffer:
            return
        
        # 计算所有轨迹的折扣回报
        all_returns = []
        all_log_probs = []
        all_values = []
        
        for trajectory in self.trajectory_buffer:
            # 计算折扣回报
            returns = []
            G = 0.0
            for reward in reversed(trajectory['rewards']):
                G = reward + self.gamma * G
                returns.insert(0, G)
            
            all_returns.extend(returns)
            all_log_probs.extend(trajectory['log_probs'])
            if self.use_baseline:
                all_values.extend(trajectory['values'])
        
        if not all_returns:
            return
        
        # 转换为张量
        returns = torch.tensor(all_returns, dtype=torch.float32)
        log_probs = torch.stack(all_log_probs)
        
        if self.use_baseline and all_values:
            # 重新计算values以避免梯度图问题
            all_states = []
            for trajectory in self.trajectory_buffer:
                all_states.extend(trajectory['states'])
            
            # 重新前向传播计算values
            values = []
            for state in all_states:
                state_tensor = self.state_to_tensor(state)
                value = self.value_net(state_tensor)
                values.append(value)
            
            values = torch.stack(values).squeeze(-1)
            
            # 计算优势函数
            advantages = returns - values
            
            # 标准化优势
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()  # 移除retain_graph=True
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            # 使用优势计算策略损失
            policy_loss = -(log_probs * advantages.detach()).mean()
        else:
            # 不使用基线
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = -(log_probs * returns).mean()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 清空缓冲区以避免重复使用
        self.trajectory_buffer.clear()
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int], List[float]]:
        """训练智能体"""
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        
        success_window = []
        
        for episode in range(n_episodes):
            reward, steps, success = self.train_episode(episode)
            self.episode_rewards.append(reward)
            self.episode_steps.append(steps)
            
            # 计算成功率
            success_window.append(1 if success else 0)
            if len(success_window) > 100:
                success_window.pop(0)
            
            current_success_rate = np.mean(success_window)
            self.success_rate.append(current_success_rate)
            
            # 学习率调度
            if episode % 500 == 0:
                self.policy_scheduler.step()
                if self.use_baseline:
                    self.value_scheduler.step()
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, "
                      f"平均步数 = {avg_steps:.2f}, 成功率 = {current_success_rate:.3f}")
        
        return self.episode_rewards, self.episode_steps, self.success_rate
    
    def test_episode(self, render: bool = False, use_exploration: bool = False) -> Tuple[float, int, List, bool]:
        """测试一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 500
        
        with torch.no_grad():
            while steps < max_steps:
                state_tensor = self.state_to_tensor(state)
                action_probs = self.policy_net(state_tensor)
                
                if use_exploration:
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    action = int(torch.argmax(action_probs))
                
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                
                if done:
                    break
                
                state = next_state
        
        success = (steps < max_steps and done)
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'use_baseline': self.use_baseline,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rate': self.success_rate
        }
        if self.use_baseline:
            save_dict['value_net'] = self.value_net.state_dict()
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        if checkpoint['use_baseline'] and hasattr(self, 'value_net'):
            self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_net.eval()
        if hasattr(self, 'value_net'):
            self.value_net.eval()


def main():
    """主函数"""
    print("=== 改进的REINFORCE算法演示 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 最大速度: {env.max_speed}")
    print(f"  - 动作数量: {env.n_actions}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    
    # 创建改进的REINFORCE智能体
    print("\n=== 改进的REINFORCE智能体（带基线）===")
    agent = ImprovedREINFORCEAgent(
        env=env,
        lr=0.003,
        gamma=0.99,
        hidden_dim=256,
        use_baseline=True
    )
    
    print(f"智能体参数：")
    print(f"  - 学习率: 0.003")
    print(f"  - 折扣因子 γ: 0.99")
    print(f"  - 隐藏层维度: 256")
    print(f"  - 使用基线: True")
    print(f"  - 状态特征维度: {agent.state_dim}")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before, success_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}, 成功 = {success_before}")
    
    # 训练智能体
    print(f"\n=== 开始训练改进的REINFORCE ===")
    n_episodes = 2000
    rewards, steps, success_rates = agent.train(n_episodes=n_episodes, verbose=True)
    
    # 分析训练结果
    print(f"\n=== 训练结果分析 ===")
    print(f"总训练回合数: {n_episodes}")
    print(f"最终100回合平均奖励: {np.mean(rewards[-100:]):.2f}")
    print(f"最终100回合平均步数: {np.mean(steps[-100:]):.2f}")
    print(f"最终100回合成功率: {np.mean(success_rates[-100:]):.3f}")
    
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
    
    # 可视化最佳测试
    print("\n=== 可视化测试 ===")
    print("运行可视化测试（显示学习到的路径）...")
    agent.test_episode(render=True)
    
    # 保存模型
    model_path = "improved_reinforce_model.pth"
    agent.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main() 