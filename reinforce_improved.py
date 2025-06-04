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
    
    def __init__(self, env: RacetrackEnv, lr=0.001, gamma=0.99, hidden_dim=256, use_baseline=True):
        self.env = env
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # 扩展状态特征维度，增加更多有用信息
        self.state_dim = 10  # 增加更多特征维度
        self.action_dim = env.n_actions
        
        # 创建策略网络，使用更大的网络
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器 - 更缓慢的衰减
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.9)
        
        # 如果使用基线，创建价值网络
        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_dim)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)
            self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=1000, gamma=0.9)
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        
        # 修改：改为单轨迹更新而不是批量更新
        self.last_trajectory = None
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """将状态转换为增强特征张量"""
        x, y, vx, vy = state
        
        # 基础特征归一化
        norm_x = x / 31.0  # 0-31范围归一化
        norm_y = y / 16.0  # 0-16范围归一化
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 计算到最近终点的距离和方向
        min_distance = float('inf')
        goal_direction_x, goal_direction_y = 0, 0
        for goal_x, goal_y in self.env.goal_positions:
            distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            if distance < min_distance:
                min_distance = distance
                # 修正方向计算
                if distance > 0:
                    goal_direction_x = (goal_x - x) / distance
                    goal_direction_y = (goal_y - y) / distance
        
        norm_distance = min_distance / 50.0  # 归一化距离
        
        # 是否在起点
        is_start = 1.0 if (x, y) in self.env.start_positions else 0.0
        
        # 是否接近终点
        near_goal = 1.0 if min_distance <= 5 else 0.0
        
        # 速度方向是否朝向目标
        velocity_alignment = 0.0
        if min_distance > 0:
            velocity_mag = np.sqrt(vx**2 + vy**2)
            if velocity_mag > 0:
                # 速度向量：向上为负x，向右为正y（根据环境定义）
                vel_x = -vx / velocity_mag  
                vel_y = vy / velocity_mag
                velocity_alignment = max(0, vel_x * goal_direction_x + vel_y * goal_direction_y)
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy,
            norm_distance, goal_direction_x, goal_direction_y, 
            is_start, near_goal, velocity_alignment
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], temperature=1.0) -> Tuple[int, torch.Tensor]:
        """根据策略选择动作，添加温度参数"""
        state_tensor = self.state_to_tensor(state)
        action_probs = self.policy_net(state_tensor)
        
        # 添加温度缩放
        if temperature != 1.0:
            action_logits = torch.log(action_probs + 1e-8) / temperature
            action_probs = F.softmax(action_logits, dim=-1)
        
        # 应用动作掩码（更宽松）
        action_probs = self._apply_action_mask(state, action_probs)
        
        # 使用概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def _apply_action_mask(self, state: Tuple[int, int, int, int], action_probs: torch.Tensor) -> torch.Tensor:
        """应用动作掩码，减少明显的错误动作（更宽松版本）"""
        x, y, vx, vy = state
        
        # 创建动作掩码
        mask = torch.ones_like(action_probs)
        
        for i, (ax, ay) in enumerate(self.env.actions):
            # 预测下一步位置
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            new_x = x - new_vx  # 向上移动
            new_y = y + new_vy  # 向右移动
            
            # 如果明显会撞墙，降低概率（但不完全禁止）
            if (new_x < 0 or new_x >= self.env.track_size[0] or 
                new_y < 0 or new_y >= self.env.track_size[1]):
                mask[i] = 0.3  # 降低但不完全禁止
            elif (new_x < self.env.track_size[0] and new_y < self.env.track_size[1] and 
                  self.env.track[new_x, new_y] == 1):
                mask[i] = 0.3  # 降低但不完全禁止
        
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
        """训练一个episode，修复探索策略"""
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
        
        # 修改：更好的探索温度调度
        temperature = max(0.8, 1.5 - episode_num / 3000)  # 更慢的探索衰减
        
        # 收集完整轨迹
        while steps < max_steps:
            action, log_prob = self.select_action(state, temperature)
            value = self.get_value(state) if self.use_baseline else torch.tensor(0.0)
            
            next_state, reward, done = self.env.step(action)
            
            # 改进的奖励塑形
            shaped_reward = self._shape_reward(state, next_state, reward, done, steps)
            
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
        
        # 判断成功
        success = (steps < max_steps and done)
        
        # 如果没有成功，给额外惩罚
        if not success:
            rewards[-1] -= 20  # 更大的失败惩罚
        
        # 存储轨迹
        self.last_trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'success': success
        }
        
        # 每个episode都更新（而不是批量更新）
        self._update_networks()
        
        return total_reward, steps, success
    
    def _shape_reward(self, state, next_state, reward, done, steps):
        """改进的奖励塑形"""
        if done and reward > 0:
            return reward + 50  # 额外成功奖励
        
        # 距离奖励
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        # 当前和下一步到最近终点的距离
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        
        # 进步奖励（更大）
        progress_reward = (curr_dist - next_dist) * 0.5
        
        # 速度奖励 - 鼓励保持合理速度
        _, _, vx, vy = next_state
        speed = vx + vy
        speed_reward = min(speed, 4) * 0.1  # 鼓励但不过度
        
        # 生存奖励 - 鼓励不要太快结束
        survival_reward = 0.01
        
        # 接近终点时的额外奖励
        proximity_bonus = 0.0
        if next_dist <= 5:
            proximity_bonus = (5 - next_dist) * 2.0
        
        return reward + progress_reward + speed_reward + survival_reward + proximity_bonus
    
    def _update_networks(self):
        """更新网络（每个episode更新一次）"""
        if not self.last_trajectory:
            return
        
        trajectory = self.last_trajectory
        
        # 计算折扣回报
        returns = []
        G = 0.0
        for reward in reversed(trajectory['rewards']):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        if not returns:
            return
        
        # 转换为张量
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(trajectory['log_probs'])
        
        if self.use_baseline and trajectory['values']:
            # 重新计算values
            values = []
            for state in trajectory['states']:
                state_tensor = self.state_to_tensor(state)
                value = self.value_net(state_tensor)
                values.append(value)
            
            values = torch.stack(values).squeeze(-1)
            
            # 计算优势函数
            advantages = returns - values
            
            # 标准化优势（如果有多个值）
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
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
            if episode % 1000 == 0 and episode > 0:
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
        lr=0.001,  # 降低学习率
        gamma=0.99,
        hidden_dim=256,
        use_baseline=True
    )
    
    print(f"智能体参数：")
    print(f"  - 学习率: 0.001")
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
    n_episodes = 3000  # 增加训练轮数
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