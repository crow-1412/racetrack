"""
REINFORCE with Baseline 强化学习智能体 - 基于Actor-Critic成功经验的优化版本

本文件基于成功的Actor-Critic算法经验对REINFORCE进行了深度优化：

核心改进：
1. 严格的动作掩码 - 从Actor-Critic借鉴的安全机制
2. 简化的奖励塑形 - 避免过度工程化 
3. 分离的优化器配置 - 策略和价值使用不同学习率
4. 极慢的探索衰减 - 防止过早收敛
5. 最佳模型保护机制 - 防止性能退化
6. 优化的状态表示 - 8维精心设计的特征

作者：AI Assistant  
最后更新：2024年
基于：Actor-Critic成功经验（60%+成功率）
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

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"🎲 优化REINFORCE随机种子已设置为: {RANDOM_SEED}")


class ImprovedPolicyNetwork(nn.Module):
    """
    改进的策略网络
    
    基于Actor-Critic成功经验的网络架构
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ImprovedPolicyNetwork, self).__init__()
        
        # 使用与成功Actor-Critic相似的架构
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """网络参数初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        shared_features = self.shared_layers(state)
        action_logits = self.policy_head(shared_features)
        return F.softmax(action_logits, dim=-1)


class ImprovedValueNetwork(nn.Module):
    """
    改进的价值网络（基线）
    
    基于Actor-Critic成功经验的网络架构
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ImprovedValueNetwork, self).__init__()
        
        # 使用与成功Actor-Critic相似的架构
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """网络参数初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        shared_features = self.shared_layers(state)
        return self.value_head(shared_features)


class OptimizedREINFORCEAgent:
    """
    基于Actor-Critic成功经验优化的REINFORCE智能体
    
    核心改进：
    1. 严格的动作掩码机制
    2. 简化的奖励塑形
    3. 分离的优化器配置
    4. 极慢的探索衰减
    5. 最佳模型保护
    """
    
    def __init__(self, env: RacetrackEnv, lr_policy: float = 0.0005, lr_value: float = 0.0003,
                 gamma: float = 0.99, hidden_dim: int = 128, entropy_coef: float = 0.05):
        """
        初始化优化的REINFORCE智能体
        
        Args:
            env: 环境
            lr_policy: 策略网络学习率（从Actor-Critic经验调整）
            lr_value: 价值网络学习率（从Actor-Critic经验调整）
            gamma: 折扣因子
            hidden_dim: 隐藏层维度
            entropy_coef: 熵正则化系数
        """
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # 状态特征维度（与成功的Actor-Critic相同）
        self.state_dim = 8
        self.action_dim = env.n_actions
        
        # 创建改进的网络
        self.policy_net = ImprovedPolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.value_net = ImprovedValueNetwork(self.state_dim, hidden_dim)
        
        # 分离的优化器（从Actor-Critic成功经验）
        self.policy_optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr_policy, 
            weight_decay=1e-5
        )
        self.value_optimizer = optim.AdamW(
            self.value_net.parameters(), 
            lr=lr_value,
            weight_decay=1e-5
        )
        
        # 优化的探索策略（固定范围，避免过度增长）
        self.epsilon = 0.3              # 适中的初始探索率
        self.epsilon_min = 0.05         # 更低的最小探索率
        self.epsilon_decay = 0.9995     # 较快但稳定的衰减
        
        # 添加奖励标准化（降低方差）
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        self.reward_alpha = 0.01
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        
        # 最佳模型保护（更严格）
        self.best_success_rate = 0.0
        self.best_model_state = None
        self.patience = 0
        self.max_patience = 150  # 降低耐心值
        self.no_improvement_count = 0
    
    def state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        状态转换为张量（使用与成功Actor-Critic完全相同的状态表示）
        """
        x, y, vx, vy = state
        
        # 1. 基础特征归一化到[0,1]范围
        norm_x = x / 31.0               
        norm_y = y / 16.0               
        norm_vx = vx / self.env.max_speed  
        norm_vy = vy / self.env.max_speed  
        
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
                    goal_direction_x = -(goal_x - x) / distance  
                    goal_direction_y = (goal_y - y) / distance   
        
        # 3. 距离归一化
        max_distance = np.sqrt(31**2 + 16**2)
        norm_distance = min_distance / max_distance
        
        # 4. 计算速度与目标方向的对齐度
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
    
    def _apply_strict_action_mask(self, state: Tuple[int, int, int, int], 
                                action_probs: torch.Tensor) -> torch.Tensor:
        """
        应用严格的动作掩码（完全从成功的Actor-Critic复制）
        
        这是关键的安全机制，确保智能体不会选择明显错误的动作
        """
        x, y, vx, vy = state
        mask = torch.ones_like(action_probs)
        
        # 遍历所有可能的动作
        for i, (ax, ay) in enumerate(self.env.actions):
            # 预测执行动作后的新速度
            new_vx = max(0, min(self.env.max_speed, vx + ax))
            new_vy = max(0, min(self.env.max_speed, vy + ay))
            
            # 处理速度为0的特殊情况
            if new_vx == 0 and new_vy == 0 and (x, y) not in self.env.start_positions:
                new_vx = 1
                new_vy = 1
            
            # 预测下一步位置
            new_x = x - new_vx  # 向上移动（x减小）
            new_y = y + new_vy  # 向右移动（y增大）
            
            # 检查是否会发生碰撞
            if self.env._check_collision(x, y, new_x, new_y):
                mask[i] = 0.0  # 禁止此动作
        
        # 确保至少有一个动作可选
        if mask.sum() == 0:
            mask.fill_(1.0)
        
        # 重新归一化概率分布
        masked_probs = action_probs * mask
        return masked_probs / (masked_probs.sum() + 1e-8)
    
    def select_action(self, state: Tuple[int, int, int, int], training: bool = True):
        """
        选择动作（改进版本，测试时更稳定）
        """
        state_tensor = self.state_to_tensor(state)
        
        if training:
            action_probs = self.policy_net(state_tensor)
        else:
            # 测试时使用确定性策略
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
        
        # 应用严格的动作掩码
        action_probs = self._apply_strict_action_mask(state, action_probs)
        
        # 🔧 修复：在起点时强制移动
        x, y, vx, vy = state
        start_positions = [(31, i) for i in range(17)]  # 起点位置
        
        if (x, y) in start_positions and vx == 0 and vy == 0:
            # 在起点且速度为零时，强制选择移动动作，禁止"停留"
            action_probs = action_probs.clone()
            action_probs[0] = 0.0  # 禁止动作0（停留）
            # 重新标准化
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                # 所有动作都被禁止时，给一个默认动作
                action_probs = torch.zeros_like(action_probs)
                action_probs[1] = 1.0  # 动作1：向前移动
        
        # 动作选择策略
        if training and random.random() < self.epsilon:
            # 训练模式探索：在有效动作中随机选择
            valid_actions = (action_probs > 0).nonzero().squeeze(-1)
            if len(valid_actions) > 0:
                action = valid_actions[random.randint(0, len(valid_actions)-1)]
            else:
                action = torch.argmax(action_probs)
        else:
            # 贪心策略：选择概率最高的动作
            action = torch.argmax(action_probs)
        
        # 计算动作的对数概率
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def _enhanced_reward_shaping(self, state, next_state, reward, done, steps):
        """
        增强的奖励塑形（进一步优化）
        
        设计原则：
        1. 更强的目标导向
        2. 更合理的进步奖励
        3. 避免局部最优
        """
        bonus = 0.0
        
        # 1. 成功/失败的明确奖励
        if done and reward > 0:
            bonus += 200    # 增加成功奖励
        elif reward == -10:  # 碰撞
            bonus -= 100    # 增加碰撞惩罚
        
        # 2. 改进的进步奖励
        x, y, vx, vy = state
        next_x, next_y, next_vx, next_vy = next_state
        
        # 计算到最近目标的欧几里得距离
        curr_dist = min([np.sqrt((x - gx)**2 + (y - gy)**2) for gx, gy in self.env.goal_positions])
        next_dist = min([np.sqrt((next_x - gx)**2 + (next_y - gy)**2) for gx, gy in self.env.goal_positions])
        
        # 距离减少奖励（更精细）
        dist_improvement = curr_dist - next_dist
        if dist_improvement > 0.5:
            bonus += 5.0 * dist_improvement
        elif dist_improvement > 0:
            bonus += 2.0 * dist_improvement
        
        # 3. 速度奖励（朝向目标的速度）
        if curr_dist > 0:
            # 找到最近的目标
            closest_goal = min(self.env.goal_positions, 
                             key=lambda g: np.sqrt((x - g[0])**2 + (y - g[1])**2))
            goal_x, goal_y = closest_goal
            
            # 计算朝向目标的速度分量
            dir_to_goal_x = -(goal_x - next_x) / max(curr_dist, 1e-6)
            dir_to_goal_y = (goal_y - next_y) / max(curr_dist, 1e-6)
            
            # 速度与目标方向的点积
            velocity_toward_goal = next_vx * dir_to_goal_x + next_vy * dir_to_goal_y
            if velocity_toward_goal > 0:
                bonus += 1.0 * velocity_toward_goal
        
        # 4. 距离惩罚（远离目标的轻微惩罚）
        if next_dist > curr_dist:
            bonus -= 1.0
        
        # 5. 步数效率奖励
        if steps < 100:  # 早期完成有额外奖励
            bonus += 0.5
        
        # 6. 更小的时间惩罚
        bonus -= 0.05
        
        return reward + bonus
    
    def _normalize_reward(self, reward: float) -> float:
        """
        奖励标准化（降低方差）
        """
        # 更新运行统计
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        
        # 更新标准差
        var = (reward - self.reward_running_mean) ** 2
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * var
        
        # 标准化
        normalized = (reward - self.reward_running_mean) / (np.sqrt(self.reward_running_std) + 1e-8)
        
        # 裁剪到合理范围
        return np.clip(normalized, -10, 10)
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """计算折扣回报"""
        returns = []
        G = 0
        
        # 从后向前计算折扣回报
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.append(G)
        
        return list(reversed(returns))
    
    def collect_episode(self, max_steps: int = 200) -> Tuple[float, int, bool, Dict]:
        """收集完整episode数据（优化版本）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_normalized_rewards = []
        
        last_reward = 0
        
        while steps < max_steps:
            action, log_prob = self.select_action(state, training=True)
            prev_state = state
            
            next_state, reward, done = self.env.step(action)
            last_reward = reward
            
            # 使用增强的奖励塑形
            shaped_reward = self._enhanced_reward_shaping(prev_state, next_state, reward, done, steps)
            
            # 奖励标准化
            normalized_reward = self._normalize_reward(shaped_reward)
            
            # 存储数据
            episode_states.append(self.state_to_tensor(prev_state))
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(shaped_reward)
            episode_normalized_rewards.append(normalized_reward)
            
            total_reward += reward  # 使用原始奖励计算回报
            steps += 1
            
            if done:
                break
                
            state = next_state
        
        # 使用标准化奖励计算折扣回报
        returns = self.compute_returns(episode_normalized_rewards)
        
        success = (steps < max_steps and done and last_reward == 100)
        
        episode_data = {
            'states': episode_states,
            'actions': episode_actions,
            'log_probs': episode_log_probs,
            'rewards': episode_normalized_rewards,
            'returns': returns
        }
        
        return total_reward, steps, success, episode_data
    
    def update_networks(self, episode_data: Dict):
        """
        更新策略和价值网络（进一步优化版本）
        """
        # 准备数据
        states = torch.stack(episode_data['states'])
        actions = torch.tensor(episode_data['actions'], dtype=torch.long)
        log_probs = torch.stack(episode_data['log_probs'])
        returns = torch.tensor(episode_data['returns'], dtype=torch.float32)
        
        # 1. 多步价值网络更新（提高基线质量）
        for _ in range(3):  # 多次更新价值网络
            values = self.value_net(states).squeeze()
            
            # 价值损失（更稳定的目标）
            value_targets = returns.detach()
            value_loss = F.mse_loss(values, value_targets)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        # 2. 策略网络更新
        # 重新计算当前策略下的动作概率
        current_action_probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(current_action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # 计算优势（使用最新的价值函数）
        with torch.no_grad():
            final_values = self.value_net(states).squeeze()
            advantages = returns - final_values
            
            # 优势标准化（更稳定）
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = torch.clamp(advantages, -3.0, 3.0)  # 更严格的裁剪
        
        # 策略损失（REINFORCE）
        policy_loss = -(new_log_probs * advantages).mean()
        
        # 熵正则化（动态调整）
        entropy = action_dist.entropy().mean()
        entropy_coef = max(0.01, self.entropy_coef * (1 - len(self.episode_rewards) / 2000))
        policy_total_loss = policy_loss - entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        policy_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 记录损失
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练单个episode"""
        # 收集完整episode
        total_reward, steps, success, episode_data = self.collect_episode()
        
        # 更新网络
        self.update_networks(episode_data)
        
        # 更新探索率（每10个episode更新一次）
        if episode_num % 10 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward, steps, success
    
    def test_episode(self, render: bool = False, debug: bool = False) -> Tuple[float, int, List, bool]:
        """测试单个episode（修正版本，与训练环境一致）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 200  # 修正：与训练时保持一致
        
        if debug:
            print(f"DEBUG: 测试开始，max_steps={max_steps}, 初始状态={state}")
        
        last_reward = 0
        collision_count = 0
        with torch.no_grad():
            while steps < max_steps:
                action, log_prob = self.select_action(state, training=False)
                
                if debug and steps < 10:
                    print(f"DEBUG: Step {steps}, state={state}, action={action}, log_prob={log_prob:.4f}")
                
                next_state, reward, done = self.env.step(action)
                
                if reward == -10:  # 碰撞
                    collision_count += 1
                
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                last_reward = reward
                
                if debug and (reward != -1 or steps % 50 == 0):
                    print(f"DEBUG: Step {steps}, reward={reward}, total_reward={total_reward}, done={done}")
                
                if done:
                    if debug:
                        print(f"DEBUG: Episode结束，steps={steps}, last_reward={last_reward}, success={last_reward==100}")
                    break
                
                state = next_state
        
        success = (steps < max_steps and done and last_reward == 100)
        
        if debug:
            print(f"DEBUG: 最终结果 - steps={steps}, max_steps={max_steps}, done={done}, last_reward={last_reward}")
            print(f"DEBUG: success判断 - (steps < max_steps)={steps < max_steps}, done={done}, (last_reward==100)={last_reward==100}")
            print(f"DEBUG: collision_count={collision_count}, total_reward={total_reward}")
        
        if render:
            self.env.render(show_path=path)
        
        return total_reward, steps, path, success
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rate': self.success_rate,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'epsilon': self.epsilon
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])


def main_ultimate_reinforce_training():
    """
    终极优化版本的REINFORCE训练函数
    """
    print("=== 终极优化版REINFORCE训练 ===")
    print(f"🎲 使用固定随机种子: {RANDOM_SEED}")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建终极优化的REINFORCE智能体
    agent = OptimizedREINFORCEAgent(
        env=env,
        lr_policy=0.0003,    # 更保守的学习率
        lr_value=0.0001,     # 更保守的价值学习率
        gamma=0.98,          # 稍微降低折扣因子
        hidden_dim=128,
        entropy_coef=0.03    # 降低熵系数
    )
    
    print(f"终极REINFORCE配置:")
    print(f"  - 策略学习率: 0.0003 (更保守)")
    print(f"  - 价值学习率: 0.0001 (更保守)")
    print(f"  - 折扣因子: 0.98 (短期奖励偏好)")
    print(f"  - 网络维度: 128")
    print(f"  - 熵正则化: 0.03 (动态调整)")
    print(f"  - 探索策略: 稳定衰减 (0.9995)")
    print(f"  - 动作掩码: 严格碰撞检测")
    print(f"  - 奖励系统: 增强塑形 + 标准化")
    print(f"  - 基线训练: 多步更新")
    
    # 训练前基准测试
    print("\n=== 训练前基准 ===")
    reward_before, steps_before, _, success_before = agent.test_episode()
    print(f"基准性能: 奖励={reward_before:.1f}, 步数={steps_before}, 成功={success_before}")
    
    # 更保守的分阶段训练
    n_episodes = 1500
    stage1_episodes = 500   # 阶段1：基础学习
    stage2_episodes = 600   # 阶段2：稳定训练  
    stage3_episodes = 400   # 阶段3：精调优化
    
    print(f"\n=== 保守分阶段训练计划 ===")
    print(f"  阶段1 (0-{stage1_episodes}): 基础策略学习")
    print(f"  阶段2 ({stage1_episodes}-{stage1_episodes+stage2_episodes}): 稳定性训练")
    print(f"  阶段3 ({stage1_episodes+stage2_episodes}-{n_episodes}): 精调优化")
    
    # 训练统计
    success_window = deque(maxlen=50)  # 更小的窗口
    reward_window = deque(maxlen=25)
    performance_window = deque(maxlen=30)
    
    for episode in range(n_episodes):
        # 保守的分阶段调整
        if episode == stage1_episodes:
            print(f"\n🔄 进入阶段2: 稳定性优先")
            for param_group in agent.policy_optimizer.param_groups:
                param_group['lr'] *= 0.8
            for param_group in agent.value_optimizer.param_groups:
                param_group['lr'] *= 0.9
                
        elif episode == stage1_episodes + stage2_episodes:
            print(f"\n🔧 进入阶段3: 精调模式")
            for param_group in agent.policy_optimizer.param_groups:
                param_group['lr'] *= 0.6
            for param_group in agent.value_optimizer.param_groups:
                param_group['lr'] *= 0.8
        
        # 训练一个episode
        reward, steps, success = agent.train_episode(episode)
        
        agent.episode_rewards.append(reward)
        agent.episode_steps.append(steps)
        success_window.append(1 if success else 0)
        reward_window.append(reward)
        
        current_success_rate = np.mean(success_window)
        agent.success_rate.append(current_success_rate)
        
        # 更稳定的性能保护机制
        if episode >= 50:
            performance_window.append(current_success_rate)
            
            # 发现新的最佳性能时保存模型
            if current_success_rate > agent.best_success_rate + 0.01:  # 需要明显改进
                agent.best_success_rate = current_success_rate
                agent.best_model_state = {
                    'policy_net': agent.policy_net.state_dict().copy(),
                    'value_net': agent.value_net.state_dict().copy(),
                    'episode': episode,
                    'success_rate': current_success_rate
                }
                agent.patience = 0
                agent.no_improvement_count = 0
                print(f"💾 保存最佳模型: Episode {episode+1}, 成功率={current_success_rate:.3f}")
            else:
                agent.patience += 1
                agent.no_improvement_count += 1
            
            # 更保守的性能退化检测
            if len(performance_window) == 30:
                recent_performance = np.mean(list(performance_window)[-15:])
                early_performance = np.mean(list(performance_window)[:15])
                
                # 只有明显退化才恢复
                if (recent_performance < early_performance * 0.6 and 
                    agent.patience > agent.max_patience and 
                    agent.best_model_state):
                    
                    print(f"\n⚠️ 检测到严重性能退化！")
                    print(f"   早期性能: {early_performance:.3f}")
                    print(f"   最近性能: {recent_performance:.3f}")
                    print(f"   恢复最佳模型...")
                    
                    agent.policy_net.load_state_dict(agent.best_model_state['policy_net'])
                    agent.value_net.load_state_dict(agent.best_model_state['value_net'])
                    print(f"   已恢复Episode {agent.best_model_state['episode']+1}的模型")
                    
                    # 适度重置探索率
                    agent.epsilon = min(0.2, agent.epsilon * 1.2)
                    print(f"   适度重置探索率至{agent.epsilon:.3f}")
                    agent.patience = 0
                    agent.no_improvement_count = 0
        
        # 定期输出训练进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(agent.episode_steps[-25:])
            
            recent_policy_loss = np.mean(agent.policy_losses[-10:]) if agent.policy_losses else 0
            recent_value_loss = np.mean(agent.value_losses[-10:]) if agent.value_losses else 0
            
            stage_name = "阶段1" if episode < stage1_episodes else "阶段2" if episode < stage1_episodes + stage2_episodes else "阶段3"
            
            print(f"{stage_name} Episode {episode + 1:4d}: "
                  f"奖励={avg_reward:6.1f}, 步数={avg_steps:5.1f}, "
                  f"成功率={current_success_rate:.3f}, ε={agent.epsilon:.3f}")
            print(f"                     策略损失={recent_policy_loss:.4f}, "
                  f"价值损失={recent_value_loss:.4f}")
            print(f"                     最佳成功率={agent.best_success_rate:.3f}, 耐心={agent.patience}")
    
    # 最终评估
    print(f"\n=== 训练完成，最终评估 ===")
    final_success = np.mean(agent.success_rate[-100:]) if len(agent.success_rate) >= 100 else 0
    print(f"最终100回合成功率: {final_success:.3f}")
    print(f"历史最佳成功率: {agent.best_success_rate:.3f}")
    
    # 如果最终性能不如历史最佳，恢复最佳模型
    if agent.best_model_state and final_success < agent.best_success_rate * 0.8:
        print(f"\n🔄 最终性能不如历史最佳，恢复最佳模型进行测试...")
        agent.policy_net.load_state_dict(agent.best_model_state['policy_net'])
        agent.value_net.load_state_dict(agent.best_model_state['value_net'])
    
    # 最终测试
    print(f"\n=== 最终测试（50次） ===")
    test_results = []
    for i in range(50):
        if (i + 1) % 10 == 0:
            print(f"测试进度: {i+1}/50")
        reward, steps, path, success = agent.test_episode()
        test_results.append((reward, steps, success))
    
    final_success_rate = np.mean([r[2] for r in test_results])
    final_avg_reward = np.mean([r[0] for r in test_results])
    final_avg_steps = np.mean([r[1] for r in test_results])
    
    print(f"\n📊 优化REINFORCE最终结果:")
    print(f"  成功率: {final_success_rate:.1%}")
    print(f"  平均奖励: {final_avg_reward:.1f}")
    print(f"  平均步数: {final_avg_steps:.1f}")
    print(f"  历史最佳训练成功率: {agent.best_success_rate:.1%}")
    
    # 与其他算法对比
    print(f"\n🏆 算法性能对比:")
    print(f"  原版PPO成功率: 10%")
    print(f"  优化PPO成功率: 26%")
    print(f"  Actor-Critic成功率: 60%+")
    print(f"  原版REINFORCE成功率: 0%")
    print(f"  优化REINFORCE成功率: {final_success_rate:.1%}")
    
    if final_success_rate > 0.5:
        print("🎉 优化REINFORCE表现优秀！接近Actor-Critic水平")
    elif final_success_rate > 0.3:
        print("✅ 优化REINFORCE表现良好，显著超越PPO算法")
    elif final_success_rate > 0.2:
        print("👍 优化REINFORCE表现不错，超越原版PPO")
    elif final_success_rate > 0.05:
        print("⚡ 优化REINFORCE有明显改进，但仍需继续调优")
    else:
        print("⚠️ 优化REINFORCE仍需更多改进")
    
    # 保存模型
    agent.save_model("models/optimized_reinforce_model.pth")
    print(f"💾 优化REINFORCE模型已保存")
    
    # 绘制训练曲线
    plot_optimized_reinforce_curves(agent)
    
    return agent, test_results


def plot_optimized_reinforce_curves(agent):
    """绘制优化REINFORCE训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 成功率曲线
    if agent.success_rate:
        axes[0, 0].plot(agent.success_rate, label='Success Rate', color='purple', linewidth=2)
        axes[0, 0].axhline(y=agent.best_success_rate, color='red', linestyle='--', 
                          label=f'Best: {agent.best_success_rate:.3f}')
        axes[0, 0].set_title('Success Rate (Optimized REINFORCE)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
    
    # 奖励曲线
    if agent.episode_rewards:
        window_size = 50
        if len(agent.episode_rewards) > window_size:
            moving_avg = np.convolve(agent.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(moving_avg, label='Moving Average', color='blue', linewidth=2)
        axes[0, 1].plot(agent.episode_rewards, alpha=0.3, label='Raw Rewards', color='lightblue')
        axes[0, 1].set_title('Episode Rewards (Optimized REINFORCE)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    
    # 策略损失
    if agent.policy_losses:
        axes[1, 0].plot(agent.policy_losses, label='Policy Loss', color='red', alpha=0.7)
        axes[1, 0].set_title('Policy Loss (Optimized REINFORCE)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # 价值损失
    if agent.value_losses:
        axes[1, 1].plot(agent.value_losses, label='Value Loss (Baseline)', color='orange', alpha=0.7)
        axes[1, 1].set_title('Value Loss (Optimized REINFORCE)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('optimized_reinforce_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 保留原版训练函数用于对比
def main_reinforce_training():
    """
    原版REINFORCE训练函数（保留用于对比）
    """
    print("=== 原版REINFORCE训练（对比用） ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建原版REINFORCE智能体
    class OriginalREINFORCEAgent:
        def __init__(self, env, lr_policy=3e-4, lr_value=1e-3, gamma=0.99, hidden_dim=128):
            # 原版实现...
            pass
    
    # 这里可以保留原版实现用于对比...
    print("原版REINFORCE保留用于性能对比")


def debug_trained_model():
    """
    调试已训练模型的性能，检查训练测试差异
    """
    print("=== 调试已训练模型 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建智能体
    agent = OptimizedREINFORCEAgent(env=env)
    
    # 尝试加载已保存的模型
    try:
        agent.load_model("models/optimized_reinforce_model.pth")
        print("✅ 成功加载模型")
    except:
        print("❌ 未找到保存的模型，使用随机初始化")
    
    print(f"当前epsilon: {agent.epsilon}")
    
    # 测试更多episode获得可靠统计
    print("\n=== 详细测试 ===")
    test_results = []
    detailed_episodes = 3  # 前3个episode显示详细信息
    total_episodes = 20    # 总共测试20次
    
    for i in range(total_episodes):
        show_debug = i < detailed_episodes
        if show_debug:
            print(f"\n--- 测试Episode {i+1} (详细) ---")
        
        reward, steps, path, success = agent.test_episode(debug=show_debug)
        test_results.append((reward, steps, success))
        
        if show_debug:
            print(f"Episode {i+1}: 奖励={reward:.1f}, 步数={steps}, 成功={success}")
            if success:
                print("🎉 成功到达终点！")
            elif steps >= 200:
                print("❌ 超过最大步数限制")
            else:
                print("💥 发生碰撞")
        else:
            # 简化显示
            status = "✅成功" if success else f"❌失败({steps}步)"
            print(f"Episode {i+1:2d}: {status}")
    
    # 详细统计分析
    success_count = sum([r[2] for r in test_results])
    success_rate = success_count / len(test_results) * 100
    
    successful_episodes = [r for r in test_results if r[2]]
    failed_episodes = [r for r in test_results if not r[2]]
    
    print(f"\n=== 测试统计 ({len(test_results)}次测试) ===")
    print(f"🎯 成功率: {success_count}/{len(test_results)} = {success_rate:.1f}%")
    
    if successful_episodes:
        avg_steps_success = sum([r[1] for r in successful_episodes]) / len(successful_episodes)
        min_steps = min([r[1] for r in successful_episodes])
        max_steps = max([r[1] for r in successful_episodes])
        print(f"✅ 成功episode: 平均{avg_steps_success:.1f}步 (范围: {min_steps}-{max_steps}步)")
    
    if failed_episodes:
        timeout_count = sum([1 for r in failed_episodes if r[1] >= 200])
        collision_count = len(failed_episodes) - timeout_count
        print(f"❌ 失败episode: {timeout_count}次超时, {collision_count}次碰撞")
    
    # 奖励统计
    rewards = [r[0] for r in test_results]
    avg_reward = sum(rewards) / len(rewards)
    print(f"💰 平均奖励: {avg_reward:.1f}")
    
    print(f"📊 详细结果: {[1 if r[2] else 0 for r in test_results]}")
    
    # 比较训练和测试的动作选择
    print("\n=== 动作选择对比 ===")
    state = env.reset()
    print(f"测试状态: {state}")
    
    # 训练模式动作选择
    action_train, log_prob_train = agent.select_action(state, training=True)
    print(f"训练模式动作: {action_train}, log_prob: {log_prob_train:.4f}")
    
    # 测试模式动作选择
    action_test, log_prob_test = agent.select_action(state, training=False)
    print(f"测试模式动作: {action_test}, log_prob: {log_prob_test:.4f}")
    
    if action_train != action_test:
        print("⚠️ 训练和测试模式选择不同的动作！")
    else:
        print("✅ 训练和测试模式选择相同动作")
    
    # 检查动作掩码
    state_tensor = agent.state_to_tensor(state)
    action_probs = agent.policy_net(state_tensor)
    masked_probs = agent._apply_strict_action_mask(state, action_probs)
    
    print(f"原始动作概率: {action_probs.detach().numpy()}")
    print(f"掩码后概率: {masked_probs.detach().numpy()}")
    print(f"有效动作数: {(masked_probs > 0).sum().item()}")
    
    return agent, test_results


def test_saved_reinforce_model(model_path: str = "models/optimized_reinforce_model.pth", 
                              test_episodes: int = 50, 
                              show_visualization: bool = True,
                              show_detailed: bool = False):
    """
    测试保存的REINFORCE模型效果
    
    Args:
        model_path: 模型文件路径
        test_episodes: 测试回合数
        show_visualization: 是否显示可视化结果
        show_detailed: 是否显示详细信息
    
    Returns:
        dict: 测试结果统计
    """
    print("=" * 60)
    print("🧪 REINFORCE模型测试")
    print("=" * 60)
    
    # 1. 创建环境和智能体
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    agent = OptimizedREINFORCEAgent(env=env)
    
    # 2. 加载模型
    try:
        agent.load_model(model_path)
        print(f"✅ 成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"❌ 未找到模型文件: {model_path}")
        print("🔄 使用随机初始化模型进行测试...")
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return None
    
    # 3. 运行测试
    print(f"\n🚀 开始测试 ({test_episodes} 回合)...")
    
    test_results = []
    successful_paths = []
    
    for i in range(test_episodes):
        # 显示进度
        if (i + 1) % 10 == 0 or i == 0:
            print(f"进度: {i+1}/{test_episodes}")
        
        # 运行测试回合
        reward, steps, path, success = agent.test_episode(render=False, debug=show_detailed and i < 3)
        
        test_results.append({
            'episode': i + 1,
            'reward': reward,
            'steps': steps,
            'success': success,
            'path': path
        })
        
        if success:
            successful_paths.append(path)
    
    # 4. 统计分析
    print("\n" + "=" * 50)
    print("📊 测试结果统计")
    print("=" * 50)
    
    # 基础统计
    total_episodes = len(test_results)
    successful_episodes = [r for r in test_results if r['success']]
    failed_episodes = [r for r in test_results if not r['success']]
    
    success_count = len(successful_episodes)
    success_rate = success_count / total_episodes * 100
    
    print(f"🎯 总测试回合: {total_episodes}")
    print(f"✅ 成功回合: {success_count}")
    print(f"❌ 失败回合: {len(failed_episodes)}")
    print(f"🏆 成功率: {success_rate:.1f}%")
    
    # 奖励统计
    all_rewards = [r['reward'] for r in test_results]
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"\n💰 奖励统计:")
    print(f"  平均奖励: {avg_reward:.1f}")
    print(f"  奖励标准差: {std_reward:.1f}")
    print(f"  最高奖励: {max(all_rewards):.1f}")
    print(f"  最低奖励: {min(all_rewards):.1f}")
    
    # 步数统计
    all_steps = [r['steps'] for r in test_results]
    avg_steps = np.mean(all_steps)
    
    print(f"\n👣 步数统计:")
    print(f"  平均步数: {avg_steps:.1f}")
    
    if successful_episodes:
        success_steps = [r['steps'] for r in successful_episodes]
        avg_success_steps = np.mean(success_steps)
        min_success_steps = min(success_steps)
        max_success_steps = max(success_steps)
        
        print(f"  成功回合平均步数: {avg_success_steps:.1f}")
        print(f"  最少成功步数: {min_success_steps}")
        print(f"  最多成功步数: {max_success_steps}")
    
    # 失败原因分析
    if failed_episodes:
        timeout_episodes = [r for r in failed_episodes if r['steps'] >= 200]
        collision_episodes = [r for r in failed_episodes if r['steps'] < 200]
        
        print(f"\n⚠️ 失败原因分析:")
        print(f"  超时失败: {len(timeout_episodes)} 回合 ({len(timeout_episodes)/total_episodes*100:.1f}%)")
        print(f"  碰撞失败: {len(collision_episodes)} 回合 ({len(collision_episodes)/total_episodes*100:.1f}%)")
    
    # 性能评级
    print(f"\n🏅 性能评级:")
    if success_rate >= 60:
        rating = "🥇 优秀 (≥60%)"
    elif success_rate >= 40:
        rating = "🥈 良好 (40-59%)"
    elif success_rate >= 20:
        rating = "🥉 一般 (20-39%)"
    elif success_rate >= 10:
        rating = "⚡ 较差 (10-19%)"
    else:
        rating = "❌ 很差 (<10%)"
    
    print(f"  {rating}")
    
    # 与其他算法对比
    print(f"\n🔍 算法对比:")
    print(f"  原版PPO: ~10%")
    print(f"  优化PPO: ~26%") 
    print(f"  Actor-Critic: ~60%")
    print(f"  当前REINFORCE: {success_rate:.1f}%")
    
    if success_rate > 60:
        print("  🎉 表现优秀！达到或超过Actor-Critic水平")
    elif success_rate > 26:
        print("  ✅ 表现良好！超过了优化PPO算法")
    elif success_rate > 10:
        print("  👍 表现不错！超过了原版PPO算法")
    else:
        print("  📈 还有改进空间")
    
    # 5. 可视化结果
    if show_visualization and successful_paths:
        print(f"\n🎨 生成可视化结果...")
        
        # 随机选择几条成功路径进行可视化
        num_paths_to_show = min(3, len(successful_paths))
        selected_paths = np.random.choice(len(successful_paths), num_paths_to_show, replace=False)
        
        for i, path_idx in enumerate(selected_paths):
            print(f"显示成功路径 {i+1}/{num_paths_to_show}")
            env.render(show_path=successful_paths[path_idx])
    
    # 6. 详细结果展示
    if show_detailed:
        print(f"\n📋 详细结果 (前10回合):")
        for i, result in enumerate(test_results[:10]):
            status = "✅成功" if result['success'] else "❌失败"
            print(f"  第{result['episode']:2d}回合: {status} | "
                  f"奖励={result['reward']:6.1f} | 步数={result['steps']:3d}")
    
    # 7. 返回统计结果
    summary = {
        'total_episodes': total_episodes,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'successful_paths': successful_paths,
        'test_results': test_results
    }
    
    print(f"\n✨ 测试完成！")
    return summary


def quick_test_reinforce():
    """
    快速测试REINFORCE模型（10回合）
    """
    print("⚡ 快速测试模式 (10回合)")
    return test_saved_reinforce_model(
        test_episodes=10, 
        show_visualization=False,
        show_detailed=True
    )


def comprehensive_test_reinforce():
    """
    全面测试REINFORCE模型（100回合）
    """
    print("🔬 全面测试模式 (100回合)")
    return test_saved_reinforce_model(
        test_episodes=100,
        show_visualization=True,
        show_detailed=False
    )


if __name__ == "__main__":
    # 首先调试已训练的模型
    print("🔍 首先调试已训练模型，检查训练测试差异...")
    debug_trained_model()
    
    print("\n" + "="*50)
    # 运行终极优化的REINFORCE训练
    # main_ultimate_reinforce_training()
    
    # 运行快速测试
    print("\n" + "="*50)
    print("🧪 运行模型测试...")
    test_saved_reinforce_model()