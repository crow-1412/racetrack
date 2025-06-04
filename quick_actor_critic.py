import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from racetrack_env import RacetrackEnv


class SimpleActorNetwork(nn.Module):
    """简化的Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(SimpleActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 更好的初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return F.softmax(action_logits, dim=-1)


class SimpleCriticNetwork(nn.Module):
    """简化的Critic网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(SimpleCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 更好的初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class QuickActorCriticAgent:
    """
    快速收敛的Actor-Critic智能体
    """
    
    def __init__(self, env: RacetrackEnv, actor_lr=0.01, critic_lr=0.02, gamma=0.95, hidden_dim=64):
        self.env = env
        self.gamma = gamma
        
        # 简化状态特征 - 只保留最重要的
        self.state_dim = 6  # 位置(2) + 速度(2) + 距离(1) + 方向(1)
        self.action_dim = env.n_actions
        
        # 创建简化的网络
        self.actor = SimpleActorNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.critic = SimpleCriticNetwork(self.state_dim, hidden_dim)
        
        # 使用更高的学习率
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        
        # 经验记录（用于curriculum learning）
        self.successful_states = set()
        self.good_actions = {}  # 记录好的状态-动作对
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """简化的状态特征"""
        x, y, vx, vy = state
        
        # 基础特征归一化
        norm_x = x / 31.0
        norm_y = y / 16.0
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        # 计算到最近终点的曼哈顿距离（更简单）
        min_distance = float('inf')
        best_goal_direction = 0.0
        for goal_x, goal_y in self.env.goal_positions:
            distance = abs(x - goal_x) + abs(y - goal_y)
            if distance < min_distance:
                min_distance = distance
                # 简化的方向特征：主要方向
                dx = goal_x - x
                dy = goal_y - y
                if abs(dx) > abs(dy):
                    best_goal_direction = 1.0 if dx > 0 else -1.0  # 主要向右或向左
                else:
                    best_goal_direction = 0.5 if dy > 0 else -0.5  # 主要向下或向上
        
        norm_distance = min_distance / 50.0
        
        return torch.tensor([
            norm_x, norm_y, norm_vx, norm_vy, norm_distance, best_goal_direction
        ], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int], epsilon=0.1) -> Tuple[int, torch.Tensor]:
        """动作选择（添加epsilon-greedy探索）"""
        state_tensor = self.state_to_tensor(state)
        action_probs = self.actor(state_tensor)
        
        # 检查是否有记录的好动作
        state_key = (state[0], state[1], state[2], state[3])
        if state_key in self.good_actions and random.random() < 0.3:  # 30%概率使用记录的好动作
            good_action = self.good_actions[state_key]
            if good_action < len(action_probs):
                # 增强记录的好动作的概率
                action_probs = action_probs.clone()
                action_probs[good_action] *= 3.0
                action_probs = action_probs / action_probs.sum()
        
        # epsilon-greedy探索
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            log_prob = torch.log(action_probs[action] + 1e-8)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action = action.item()
        
        return action, log_prob
    
    def get_value(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """获取状态价值"""
        state_tensor = self.state_to_tensor(state)
        return self.critic(state_tensor)
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练一个episode"""
        state = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        max_steps = 300  # 减少最大步数，强制更快决策
        
        # 动态调整探索率
        epsilon = max(0.05, 0.5 - episode_num / 1000)
        
        trajectory = []  # 记录轨迹用于后续分析
        
        while steps < max_steps:
            # 选择动作
            action, log_prob = self.select_action(state, epsilon)
            
            # 获取当前状态价值
            current_value = self.get_value(state)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            
            # 改进的奖励塑形
            shaped_reward = self._enhanced_reward_shaping(state, next_state, reward, done, steps)
            
            # 记录轨迹
            trajectory.append((state, action, shaped_reward, next_state, done))
            
            total_reward += reward
            steps += 1
            
            # 获取下一状态价值
            if done:
                next_value = torch.tensor(0.0)
            else:
                next_value = self.get_value(next_state)
            
            # 计算TD误差和更新网络
            td_target = shaped_reward + self.gamma * next_value
            td_error = td_target - current_value
            
            # 更新Critic
            critic_loss = td_error.pow(2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            
            # 更新Actor
            actor_loss = -(log_prob * td_error.detach())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            if done:
                break
            
            state = next_state
        
        success = (steps < max_steps and done)
        
        # 如果成功，记录好的状态-动作对
        if success:
            self._record_successful_trajectory(trajectory)
        
        return total_reward, steps, success
    
    def _enhanced_reward_shaping(self, state, next_state, reward, done, steps):
        """增强的奖励塑形"""
        if done and reward > 0:
            return reward + 100  # 大幅增加成功奖励
        
        x, y, _, _ = state
        next_x, next_y, _, _ = next_state
        
        # 距离奖励（更大的系数）
        curr_dist = min([abs(x - gx) + abs(y - gy) for gx, gy in self.env.goal_positions])
        next_dist = min([abs(next_x - gx) + abs(next_y - gy) for gx, gy in self.env.goal_positions])
        progress_reward = (curr_dist - next_dist) * 2.0  # 增大进步奖励
        
        # 接近终点的大幅奖励
        proximity_bonus = 0.0
        if next_dist <= 3:
            proximity_bonus = (3 - next_dist) * 10.0
        elif next_dist <= 8:
            proximity_bonus = (8 - next_dist) * 2.0
        
        # 速度奖励
        _, _, vx, vy = next_state
        speed_bonus = min(vx + vy, 3) * 0.5
        
        # 避免过长的episode
        time_penalty = -0.02 if steps > 200 else 0.0
        
        return reward + progress_reward + proximity_bonus + speed_bonus + time_penalty
    
    def _record_successful_trajectory(self, trajectory):
        """记录成功轨迹的好动作"""
        # 记录轨迹中后半段的状态-动作对（更可能是好的）
        start_record = len(trajectory) // 2
        for i in range(start_record, len(trajectory)):
            state, action, reward, next_state, done = trajectory[i]
            state_key = (state[0], state[1], state[2], state[3])
            self.good_actions[state_key] = action
            
            # 只保留最近的N个好动作，避免内存过大
            if len(self.good_actions) > 1000:
                # 随机删除一些旧的记录
                keys_to_remove = random.sample(list(self.good_actions.keys()), 200)
                for key in keys_to_remove:
                    del self.good_actions[key]
    
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
            if len(success_window) > 50:  # 用更小的窗口观察成功率
                success_window.pop(0)
            
            current_success_rate = np.mean(success_window)
            self.success_rate.append(current_success_rate)
            
            if verbose and (episode + 1) % 50 == 0:  # 更频繁的输出
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_steps = np.mean(self.episode_steps[-50:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, "
                      f"平均步数 = {avg_steps:.2f}, 成功率 = {current_success_rate:.3f}, "
                      f"好动作记录数 = {len(self.good_actions)}")
        
        return self.episode_rewards, self.episode_steps, self.success_rate
    
    def test_episode(self, render: bool = False) -> Tuple[float, int, List, bool]:
        """测试一个episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 300
        
        with torch.no_grad():
            while steps < max_steps:
                state_tensor = self.state_to_tensor(state)
                action_probs = self.actor(state_tensor)
                action = int(torch.argmax(action_probs))  # 贪婪策略
                
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


def main():
    """主函数"""
    print("=== 快速收敛的Actor-Critic算法演示 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 最大速度: {env.max_speed}")
    print(f"  - 动作数量: {env.n_actions}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    
    # 创建快速收敛的Actor-Critic智能体
    print("\n=== 快速收敛的Actor-Critic智能体 ===")
    agent = QuickActorCriticAgent(
        env=env,
        actor_lr=0.01,   # 更高的学习率
        critic_lr=0.02,  # 更高的学习率
        gamma=0.95,      # 稍微降低折扣因子
        hidden_dim=64    # 更小的网络
    )
    
    print(f"智能体参数：")
    print(f"  - Actor学习率: 0.01（高）")
    print(f"  - Critic学习率: 0.02（高）")
    print(f"  - 折扣因子 γ: 0.95")
    print(f"  - 隐藏层维度: 64（小）")
    print(f"  - 状态特征维度: {agent.state_dim}（简化）")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before, success_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}, 成功 = {success_before}")
    
    # 快速训练
    print(f"\n=== 开始快速训练 ===")
    n_episodes = 1000  # 减少训练回合，应该更快看到效果
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
        print(f"测试 {i+1}: 奖励={reward:.1f}, 步数={steps}, 成功={'是' if success else '否'}")
    
    avg_test_reward = np.mean([r[0] for r in test_results])
    avg_test_steps = np.mean([r[1] for r in test_results])
    test_success_rate = np.mean([r[2] for r in test_results])
    
    print(f"\n测试结果（10次平均）:")
    print(f"  - 平均奖励: {avg_test_reward:.2f}")
    print(f"  - 平均步数: {avg_test_steps:.2f}")
    print(f"  - 成功率: {test_success_rate:.3f}")
    
    # 性能对比
    print(f"\n=== 性能提升 ===")
    print(f"奖励提升: {avg_test_reward - reward_before:.2f}")
    print(f"步数变化: {avg_test_steps - steps_before:.0f}")
    print(f"成功率: {test_success_rate:.3f}")
    
    # 显示最佳路径
    if test_success_rate > 0:
        print("\n=== 展示成功路径 ===")
        agent.test_episode(render=True)


if __name__ == "__main__":
    main() 