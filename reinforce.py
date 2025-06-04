import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict
import random
from racetrack_env import RacetrackEnv


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return F.softmax(action_logits, dim=-1)


class ValueNetwork(nn.Module):
    """价值网络（用于Actor-Critic）"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class REINFORCEAgent:
    """
    改进的REINFORCE智能体
    使用蒙特卡洛策略梯度方法，包含基线减少方差
    """
    
    def __init__(self, env: RacetrackEnv, lr=0.001, gamma=0.95, hidden_dim=128, use_baseline=True):
        self.env = env
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # 状态特征维度：位置(2) + 速度(2) = 4
        self.state_dim = 4
        self.action_dim = env.n_actions
        
        # 创建策略网络
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 如果使用基线，创建价值网络
        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_dim)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 训练统计（参考Q-learning和Sarsa-lambda的结构）
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """将状态转换为张量"""
        x, y, vx, vy = state
        # 归一化位置和速度
        norm_x = x / self.env.track_size[0]
        norm_y = y / self.env.track_size[1]
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        return torch.tensor([norm_x, norm_y, norm_vx, norm_vy], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, torch.Tensor]:
        """根据策略选择动作"""
        state_tensor = self.state_to_tensor(state)
        action_probs = self.policy_net(state_tensor)
        
        # 使用概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def get_value(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """获取状态价值（如果使用基线）"""
        if self.use_baseline:
            state_tensor = self.state_to_tensor(state)
            return self.value_net(state_tensor)
        else:
            return torch.tensor(0.0)
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个episode"""
        state = self.env.reset()
        
        # 存储轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        total_reward = 0.0
        steps = 0
        
        # 收集完整轨迹
        while True:
            action, log_prob = self.select_action(state)
            value = self.get_value(state) if self.use_baseline else torch.tensor(0.0)
            
            next_state, reward, done = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        # 计算折扣回报
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 转换为张量
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        
        if self.use_baseline:
            values = torch.stack(values).squeeze()
            # 计算优势函数
            advantages = returns - values
            
            # 标准化优势（减少方差）
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # 使用优势计算策略损失
            policy_loss = []
            for log_prob, advantage in zip(log_probs, advantages.detach()):
                policy_loss.append(-log_prob * advantage)
        else:
            # 不使用基线，直接使用回报
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return total_reward, steps
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int]]:
        """训练智能体（参考Q-learning和Sarsa-lambda的训练方法）"""
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
    
    def test_episode(self, render: bool = False, use_exploration: bool = False) -> Tuple[float, int, List]:
        """测试一个episode（参考Q-learning和Sarsa-lambda的测试方法）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]  # 记录路径（只记录位置）
        
        with torch.no_grad():
            while steps < 1000:  # 防止无限循环
                state_tensor = self.state_to_tensor(state)
                action_probs = self.policy_net(state_tensor)
                
                if use_exploration:
                    # 使用概率采样
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    # 贪婪选择（选择概率最大的动作）
                    action = int(torch.argmax(action_probs))
                
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
    
    def get_action_probabilities(self, state: Tuple[int, int, int, int]) -> np.ndarray:
        """获取给定状态下的动作概率分布"""
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            action_probs = self.policy_net(state_tensor)
            return action_probs.numpy()
    
    def get_policy(self) -> Dict[Tuple[int, int, int, int], int]:
        """获取当前策略（贪婪策略）- 参考Q-learning方法"""
        policy: Dict[Tuple[int, int, int, int], int] = {}
        
        # 为了生成策略，我们需要遍历可能的状态
        # 这里简化实现，只返回测试过程中遇到的状态
        print("注意：策略梯度方法不能直接枚举所有状态的策略")
        return policy
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'use_baseline': self.use_baseline
        }
        if self.use_baseline:
            save_dict['value_net'] = self.value_net.state_dict()
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        if checkpoint['use_baseline'] and hasattr(self, 'value_net'):
            self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_net.eval()
        if hasattr(self, 'value_net'):
            self.value_net.eval()


class ActorCriticAgent:
    """
    Actor-Critic智能体
    结合策略梯度和时序差分学习
    """
    
    def __init__(self, env: RacetrackEnv, lr=0.001, gamma=0.95, hidden_dim=128):
        self.env = env
        self.gamma = gamma
        
        # 状态特征维度：位置(2) + 速度(2) = 4
        self.state_dim = 4
        self.action_dim = env.n_actions
        
        # 创建Actor（策略网络）和Critic（价值网络）
        self.actor = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.critic = ValueNetwork(self.state_dim, hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 训练统计（参考Q-learning和Sarsa-lambda的结构）
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """将状态转换为张量"""
        x, y, vx, vy = state
        # 归一化位置和速度
        norm_x = x / self.env.track_size[0]
        norm_y = y / self.env.track_size[1]
        norm_vx = vx / self.env.max_speed
        norm_vy = vy / self.env.max_speed
        
        return torch.tensor([norm_x, norm_y, norm_vx, norm_vy], dtype=torch.float32)
    
    def select_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, torch.Tensor]:
        """根据策略选择动作"""
        state_tensor = self.state_to_tensor(state)
        action_probs = self.actor(state_tensor)
        
        # 使用概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def get_value(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """获取状态价值"""
        state_tensor = self.state_to_tensor(state)
        return self.critic(state_tensor)
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个episode（使用TD学习更新Critic）"""
        state = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        
        while True:
            # 选择动作
            action, log_prob = self.select_action(state)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # 计算TD误差（类似Sarsa-lambda中的delta）
            value_current = self.get_value(state)
            value_next = self.get_value(next_state) if not done else torch.tensor(0.0)
            td_target = reward + self.gamma * value_next
            td_error = td_target - value_current
            
            # 更新Critic（价值网络）
            critic_loss = td_error.pow(2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 更新Actor（策略网络）
            # 使用TD误差作为优势估计
            actor_loss = -log_prob * td_error.detach()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            if done:
                break
            
            state = next_state
        
        return total_reward, steps
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int]]:
        """训练智能体（参考Q-learning和Sarsa-lambda的训练方法）"""
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
    
    def test_episode(self, render: bool = False, use_exploration: bool = False) -> Tuple[float, int, List]:
        """测试一个episode（参考Q-learning和Sarsa-lambda的测试方法）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]  # 记录路径（只记录位置）
        
        with torch.no_grad():
            while steps < 1000:  # 防止无限循环
                state_tensor = self.state_to_tensor(state)
                action_probs = self.actor(state_tensor)
                
                if use_exploration:
                    # 使用概率采样
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    # 贪婪选择（选择概率最大的动作）
                    action = int(torch.argmax(action_probs))
                
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
    
    def get_action_probabilities(self, state: Tuple[int, int, int, int]) -> np.ndarray:
        """获取给定状态下的动作概率分布"""
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            action_probs = self.actor(state_tensor)
            return action_probs.numpy()
    
    def get_state_value(self, state: Tuple[int, int, int, int]) -> float:
        """获取状态价值"""
        with torch.no_grad():
            value = self.get_value(state)
            return value.item()
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor.eval()
        self.critic.eval()


def main():
    """
    主函数：演示REINFORCE和Actor-Critic智能体的训练和测试过程
    参考Q-learning和Sarsa-lambda的主函数结构
    """
    print("=== 策略梯度方法演示 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 最大速度: {env.max_speed}")
    print(f"  - 动作数量: {env.n_actions}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    
    # 选择算法
    print("\n=== 可用算法 ===")
    print("1. REINFORCE（无基线）")
    print("2. REINFORCE（带基线）")
    print("3. Actor-Critic")
    
    algorithm = input("请选择算法 (1/2/3): ").strip()
    
    if algorithm == "1":
        print("\n=== REINFORCE（无基线）智能体 ===")
        agent = REINFORCEAgent(
            env=env,
            lr=0.002,
            gamma=0.95,
            hidden_dim=128,
            use_baseline=False
        )
        agent_name = "REINFORCE（无基线）"
    elif algorithm == "2":
        print("\n=== REINFORCE（带基线）智能体 ===")
        agent = REINFORCEAgent(
            env=env,
            lr=0.002,
            gamma=0.95,
            hidden_dim=128,
            use_baseline=True
        )
        agent_name = "REINFORCE（带基线）"
    elif algorithm == "3":
        print("\n=== Actor-Critic智能体 ===")
        agent = ActorCriticAgent(
            env=env,
            lr=0.002,
            gamma=0.95,
            hidden_dim=128
        )
        agent_name = "Actor-Critic"
    else:
        print("无效选择，使用默认的Actor-Critic算法")
        agent = ActorCriticAgent(
            env=env,
            lr=0.002,
            gamma=0.95,
            hidden_dim=128
        )
        agent_name = "Actor-Critic"
    
    print(f"\n智能体参数：")
    print(f"  - 学习率: 0.002")
    print(f"  - 折扣因子 γ: 0.95")
    print(f"  - 隐藏层维度: 128")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}")
    print(f"是否到达终点: {'是' if path_before[-1] in env.goal_positions else '否'}")
    
    # 训练智能体
    print(f"\n=== 开始训练 {agent_name} ===")
    n_episodes = 1500
    rewards, steps = agent.train(n_episodes=n_episodes, verbose=True)
    
    # 分析训练结果
    print(f"\n=== 训练结果分析 ===")
    print(f"总训练回合数: {n_episodes}")
    print(f"最终100回合平均奖励: {np.mean(rewards[-100:]):.2f}")
    print(f"最终100回合平均步数: {np.mean(steps[-100:]):.2f}")
    
    # 训练后测试
    print("\n=== 训练后测试 ===")
    reward_after, steps_after, path_after = agent.test_episode()
    print(f"训练后性能: 奖励 = {reward_after:.2f}, 步数 = {steps_after}")
    print(f"是否到达终点: {'是' if path_after[-1] in env.goal_positions else '否'}")
    
    # 性能对比
    print(f"\n=== 性能提升 ===")
    print(f"奖励提升: {reward_after - reward_before:.2f}")
    print(f"步数变化: {steps_after - steps_before}")
    
    # 可视化测试
    print("\n=== 可视化测试 ===")
    print("运行可视化测试（显示学习到的路径）...")
    agent.test_episode(render=True)
    
    # 保存模型
    model_path = f"{agent_name.lower().replace('（', '_').replace('）', '').replace(' ', '_')}_model.pth"
    agent.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main() 