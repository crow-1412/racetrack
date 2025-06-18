"""
稳定版 Gumbel-Softmax PPO - 赛车轨道问题
修复训练不稳定问题，优化温度调度和探索策略

关键改进：
1. 自适应温度调度
2. 成功率反馈机制
3. 更稳定的奖励塑形
4. 早停和模型保护
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
from racetrack_env import RacetrackEnv

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f"🔧 稳定版Gumbel-Softmax PPO随机种子已设置为: {RANDOM_SEED}")

class StableGumbelPPONetwork(nn.Module):
    """稳定版Gumbel-Softmax PPO网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(StableGumbelPPONetwork, self).__init__()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout提高稳定性
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor头部
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic头部
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        return action_logits, value


# 继承之前的Buffer类
class StableGumbelPPOBuffer:
    """稳定版经验缓冲区"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def size(self):
        return len(self.states)
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float, next_value: float = 0):
        """稳定的GAE计算"""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array([v.detach().item() if isinstance(v, torch.Tensor) else v for v in self.values], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)
        
        # 奖励裁剪提高稳定性
        rewards = np.clip(rewards, -100, 200)
        
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
        
        returns = advantages + values
        
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


class StableGumbelPPORacetrackAgent:
    """
    稳定版Gumbel-Softmax PPO智能体
    解决训练不稳定和性能退化问题
    """
    
    def __init__(self, env: RacetrackEnv, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 ppo_epochs: int = 4, batch_size: int = 128,
                 buffer_size: int = 1024, hidden_dim: int = 128):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # 状态和动作维度
        self.state_dim = 8
        self.action_dim = env.n_actions
        
        # 创建稳定版网络
        self.network = StableGumbelPPONetwork(self.state_dim, self.action_dim, hidden_dim)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4, eps=1e-5)
        
        # 经验缓冲区
        self.buffer = StableGumbelPPOBuffer(buffer_size)
        
        # 🔧 改进的温度调度
        self.temperature = 1.5      # 降低初始温度
        self.min_temperature = 0.8  # 提高最小温度
        self.temperature_decay = 0.998  # 更慢的衰减
        
        # 🔧 自适应温度调整
        self.success_rate_history = deque(maxlen=20)
        self.last_success_rate = 0.0
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rate: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        
        # 🔧 最佳模型保护 - 加强版
        self.best_success_rate = 0.0
        self.best_model_state = None
        self.patience = 0
        self.max_patience = 30  # 减少耐心值，更早干预
        
        # 🔧 新增：性能急剧下降检测
        self.performance_drop_threshold = 0.15  # 成功率下降15%触发回溯
        self.recent_success_rates = deque(maxlen=10)  # 最近10个episode的成功率
        
        # 🔧 新增：动态学习率调整
        self.initial_lr = 3e-4
        self.lr_decay_factor = 0.8
        self.lr_recovery_factor = 1.2
        
        # 🔧 新增：连续性能下降检测
        self.consecutive_rollbacks = 0  # 连续回溯次数
        self.max_consecutive_rollbacks = 3  # 最大连续回溯次数
        self.early_stop_triggered = False  # 提前停止标志
        
        # 奖励塑形参数
        self.last_distance_to_goal = None
        
        print(f"🔧 稳定版参数: 温度={self.temperature}, 最小={self.min_temperature}, 衰减={self.temperature_decay}")
    
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
    
    def gumbel_softmax_sample(self, logits: torch.Tensor, temperature: float = 1.0, hard: bool = False):
        """稳定版Gumbel-Softmax采样"""
        # 数值稳定性改进
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8, 1.0)))
        gumbel_logits = (logits + gumbel_noise) / max(temperature, 0.1)  # 防止温度过小
        
        soft_action = F.softmax(gumbel_logits, dim=-1)
        
        if hard:
            discrete_action = torch.argmax(soft_action, dim=-1)
            hard_action = F.one_hot(discrete_action, self.action_dim).float()
            # 直通估计器
            soft_action = hard_action.detach() + soft_action - soft_action.detach()
        
        return soft_action
    
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
                mask[i] = -1e9
        
        masked_logits = action_logits + mask
        
        if torch.all(mask == -1e9):
            mask.fill_(0)
            masked_logits = action_logits
        
        return masked_logits
    
    def select_action(self, state: Tuple[int, int, int, int], training: bool = True):
        """选择动作"""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits, value = self.network(state_tensor)
            
            # 应用动作掩码
            masked_logits = self.apply_action_mask(state, action_logits)
            
            if training:
                # 训练时：使用Gumbel-Softmax采样
                soft_action = self.gumbel_softmax_sample(masked_logits, self.temperature, hard=True)
                discrete_action = torch.argmax(soft_action)
                
                # 计算log概率
                action_dist = Categorical(logits=masked_logits)
                log_prob = action_dist.log_prob(discrete_action)
            else:
                # 测试时：贪婪策略
                discrete_action = torch.argmax(masked_logits)
                
                action_dist = Categorical(logits=masked_logits)
                log_prob = action_dist.log_prob(discrete_action)
        
        return discrete_action.item(), log_prob, value.squeeze()
    
    def reward_shaping(self, prev_state, state, next_state, reward, done, steps):
        """稳定的奖励塑形"""
        shaped_reward = reward
        
        x, y, vx, vy = state
        
        # 前进奖励
        current_distance = min([np.sqrt((x - gx)**2 + (y - gy)**2) 
                               for gx, gy in self.env.goal_positions])
        
        if self.last_distance_to_goal is not None:
            progress = self.last_distance_to_goal - current_distance
            if progress > 0:
                shaped_reward += progress * 0.1  # 适中的前进奖励
        
        self.last_distance_to_goal = current_distance
        
        # 适中的步数惩罚
        shaped_reward -= 0.005
        
        # 成功奖励
        if done and reward == 100:
            shaped_reward += 30  # 适中的成功奖励
        elif done and reward == -10:
            shaped_reward -= 10
        
        return shaped_reward
    
    def adaptive_temperature_update(self, current_success_rate: float):
        """🔧 自适应温度调整 - 加强版"""
        self.success_rate_history.append(current_success_rate)
        
        if len(self.success_rate_history) >= 10:
            # 计算成功率趋势
            recent_rate = np.mean(list(self.success_rate_history)[-5:])
            older_rate = np.mean(list(self.success_rate_history)[-10:-5])
            
            # 如果成功率下降，增加温度（增加探索）
            if recent_rate < older_rate - 0.05:
                self.temperature = min(2.0, self.temperature * 1.05)  # 更积极的温度增加
                print(f"    🔥 成功率下降，增加探索: T={self.temperature:.3f}")
            # 如果成功率稳定提升，逐渐降低温度
            elif recent_rate > older_rate + 0.02:
                self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
    
    def detect_performance_drop(self, current_success_rate: float) -> bool:
        """🔧 新增：检测性能急剧下降"""
        self.recent_success_rates.append(current_success_rate)
        
        if len(self.recent_success_rates) >= 5:
            # 计算最近5个episode的平均成功率
            recent_avg = np.mean(list(self.recent_success_rates)[-5:])
            # 与最佳成功率比较
            if self.best_success_rate > 0 and (self.best_success_rate - recent_avg) > self.performance_drop_threshold:
                return True
        return False
    
    def adjust_learning_rate(self, factor: float):
        """🔧 新增：动态调整学习率"""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * factor
        # 限制学习率范围
        new_lr = max(1e-5, min(1e-3, new_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        print(f"    📈 学习率调整: {current_lr:.2e} -> {new_lr:.2e}")
        return new_lr
    
    def collect_trajectory(self, max_steps: int = 300) -> Tuple[float, int, bool]:
        """收集轨迹"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        self.last_distance_to_goal = None
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(state, training=True)
            prev_state = state
            
            next_state, reward, done = self.env.step(action)
            
            # 奖励塑形
            shaped_reward = self.reward_shaping(prev_state, state, next_state, reward, done, steps)
            
            # 存储经验
            self.buffer.add(
                self.state_to_tensor(prev_state),
                action,
                shaped_reward,
                value,
                log_prob,
                done
            )
            
            total_reward += reward
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
        """稳定的策略更新"""
        if self.buffer.size() < self.batch_size:
            return
        
        # 优势归一化
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        update_count = 0
        
        # PPO更新
        for epoch in range(self.ppo_epochs):
            batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = \
                self.buffer.get_batch(self.batch_size)
            
            batch_advantages = advantages[:len(batch_advantages)]
            
            # 前向传播
            action_logits, values = self.network(batch_states)
            
            # 重新计算动作概率
            action_dist = Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(batch_actions)
            
            # 重要性采样比率
            ratio = torch.exp(log_probs - batch_old_log_probs.detach())
            
            # PPO损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            batch_returns_tensor = torch.tensor(self.buffer.returns[:len(batch_returns)], dtype=torch.float32)
            value_loss = F.mse_loss(values.squeeze(), batch_returns_tensor)
            
            # 熵奖励
            entropy = action_dist.entropy().mean()
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # 检查数值稳定性
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("⚠️ 检测到数值不稳定，跳过此次更新")
                continue
            
            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            update_count += 1
        
        # 记录损失
        if update_count > 0:
            self.policy_losses.append(total_policy_loss / update_count)
            self.value_losses.append(total_value_loss / update_count)
        
        # 清空缓冲区
        self.buffer.clear()
    
    def train_episode(self, episode_num: int) -> Tuple[float, int, bool]:
        """训练单个episode"""
        reward, steps, success = self.collect_trajectory()
        self.update_policy()
        return reward, steps, success
    
    def test_episode(self, render: bool = False, debug: bool = False) -> Tuple[float, int, List, bool]:
        """测试单个episode - 添加调试功能"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]
        max_steps = 300
        
        # 🔧 调试：重置奖励塑形状态（与训练保持一致）
        self.last_distance_to_goal = None
        
        self.network.eval()
        with torch.no_grad():
            while steps < max_steps:
                if debug and steps < 10:
                    # 比较训练和测试时的动作选择
                    action_train, _, _ = self.select_action(state, training=True)
                    action_test, _, _ = self.select_action(state, training=False)
                    print(f"Step {steps}: 训练动作={action_train}, 测试动作={action_test}, 相同={action_train==action_test}")
                
                action, _, _ = self.select_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                path.append(next_state[:2])
                
                if debug and (reward != -1 or steps % 50 == 0):
                    print(f"Step {steps}: 动作={action}, 奖励={reward}, 累计奖励={total_reward}")
                
                if done:
                    break
                
                state = next_state
        
        self.network.train()
        success = (done and reward == 100)
        
        if debug:
            print(f"测试结果: 步数={steps}, 成功={success}, 最终奖励={reward}")
        
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
            'temperature': self.temperature
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.network.load_state_dict(checkpoint['network'])


def main_stable_gumbel_ppo_training():
    """稳定版Gumbel-Softmax PPO主训练函数"""
    print("=== 稳定版 Gumbel-Softmax PPO赛车轨道训练 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 创建稳定版智能体
    agent = StableGumbelPPORacetrackAgent(
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=4,
        batch_size=128,
        buffer_size=1024,
        hidden_dim=128
    )
    
    print(f"稳定版配置:")
    print(f"  - 自适应温度调度")
    print(f"  - 最佳模型保护")
    print(f"  - 数值稳定性检查")
    print(f"  - 成功率反馈机制")
    
    # 训练设置
    n_episodes = 1000
    
    print(f"\n=== 开始稳定版训练 ===")
    
    # 训练统计
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=50)
    
    for episode in range(n_episodes):
        reward, steps, success = agent.train_episode(episode)
        
        agent.episode_rewards.append(reward)
        agent.episode_steps.append(steps)
        success_window.append(1 if success else 0)
        reward_window.append(reward)
        
        current_success_rate = np.mean(success_window)
        agent.success_rate.append(current_success_rate)
        
        # 🔧 自适应温度调整
        if episode >= 20:
            agent.adaptive_temperature_update(current_success_rate)
        
        # 🔧 最佳模型保护 - 更频繁保存
        if episode >= 30 and current_success_rate > agent.best_success_rate:
            agent.best_success_rate = current_success_rate
            agent.best_model_state = {
                'network': agent.network.state_dict().copy(),
                'optimizer': agent.optimizer.state_dict().copy(),
                'episode': episode,
                'success_rate': current_success_rate,
                'temperature': agent.temperature
            }
            agent.patience = 0
            agent.consecutive_rollbacks = 0  # 🔧 重置连续回溯计数器，因为有改进
            print(f"💾 保存最佳模型: Episode {episode+1}, 成功率={current_success_rate:.3f}")
        else:
            agent.patience += 1
        
        # 🔧 新增：性能急剧下降检测（立即干预）
        if episode >= 50:
            performance_dropped = agent.detect_performance_drop(current_success_rate)
            if performance_dropped and agent.best_model_state:
                agent.consecutive_rollbacks += 1
                print(f"\n🚨 检测到性能急剧下降! 当前: {current_success_rate:.3f}, 最佳: {agent.best_success_rate:.3f}")
                print(f"   连续回溯次数: {agent.consecutive_rollbacks}/{agent.max_consecutive_rollbacks}")
                
                # 检查是否需要提前停止
                if agent.consecutive_rollbacks >= agent.max_consecutive_rollbacks:
                    print(f"\n🛑 连续{agent.max_consecutive_rollbacks}次性能下降，触发提前停止！")
                    print(f"   模型可能已达到性能瓶颈，建议结束训练")
                    agent.early_stop_triggered = True
                    break
                
                print(f"   第{agent.consecutive_rollbacks}次回溯，恢复Episode {agent.best_model_state['episode']+1}的最佳模型...")
                
                # 恢复最佳模型
                agent.network.load_state_dict(agent.best_model_state['network'])
                agent.optimizer.load_state_dict(agent.best_model_state['optimizer'])
                agent.temperature = agent.best_model_state['temperature']
                
                # 调整学习率和温度
                agent.adjust_learning_rate(agent.lr_decay_factor)  # 降低学习率
                agent.temperature = min(2.0, agent.temperature * 1.2)  # 增加探索
                
                # 重置计数器
                agent.patience = 0
                agent.recent_success_rates.clear()
                
                print(f"   已恢复最佳状态，新温度: {agent.temperature:.3f}")
                continue
        
        # 🔧 原有的耐心值退化检测
        if agent.patience > agent.max_patience and agent.best_model_state:
            agent.consecutive_rollbacks += 1
            print(f"\n⚠️ 性能停滞 ({agent.patience}个episode无改进)，恢复最佳模型...")
            print(f"   连续回溯次数: {agent.consecutive_rollbacks}/{agent.max_consecutive_rollbacks}")
            
            # 检查是否需要提前停止
            if agent.consecutive_rollbacks >= agent.max_consecutive_rollbacks:
                print(f"\n🛑 连续{agent.max_consecutive_rollbacks}次性能停滞，触发提前停止！")
                print(f"   模型可能已达到性能瓶颈，建议结束训练")
                agent.early_stop_triggered = True
                break
            
            print(f"   第{agent.consecutive_rollbacks}次回溯，恢复Episode {agent.best_model_state['episode']+1}的模型")
            agent.network.load_state_dict(agent.best_model_state['network'])
            agent.optimizer.load_state_dict(agent.best_model_state['optimizer'])
            
            # 更保守的调整
            agent.adjust_learning_rate(agent.lr_recovery_factor)  # 恢复学习率
            agent.temperature = min(2.0, agent.temperature * 1.1)  # 轻微增加探索
            agent.patience = 0
        
        # 定期输出
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(agent.episode_steps[-25:])
            
            print(f"Episode {episode + 1:4d}: "
                  f"奖励={avg_reward:6.1f}, 步数={avg_steps:5.1f}, "
                  f"成功率={current_success_rate:.3f}, 温度={agent.temperature:.3f}")
            print(f"                     最佳成功率={agent.best_success_rate:.3f}, 耐心值={agent.patience}")
    
    # 检查训练结束原因
    if agent.early_stop_triggered:
        print(f"\n🔚 训练因连续性能下降而提前停止")
        print(f"   最佳成功率: {agent.best_success_rate:.3f} (Episode {agent.best_model_state['episode']+1})")
    else:
        print(f"\n✅ 训练正常完成")
    
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
    
    print(f"稳定版Gumbel-Softmax PPO最终结果（50次测试）:")
    print(f"  成功率: {final_success_rate:.1%}")
    print(f"  平均奖励: {final_avg_reward:.1f}")
    print(f"  平均步数: {final_avg_steps:.1f}")
    
    # 对比结果
    print(f"\n📊 连续化方法对比:")
    print(f"  高斯映射PPO成功率:    3% (梯度断裂)")
    print(f"  普通Gumbel PPO成功率: 36%->1% (训练不稳定)")
    print(f"  稳定Gumbel PPO成功率: {final_success_rate:.1%} (问题解决)")
    
    if final_success_rate > 0.4:
        print("🎉 连续化问题完全解决！")
    elif final_success_rate > 0.2:
        print("✅ 连续化显著改善")
    else:
        print("⚠️ 仍需进一步优化")
    
    # 保存模型
    agent.save_model("models/stable_gumbel_ppo_model.pth")
    print(f"稳定版Gumbel-Softmax PPO模型已保存")
    
    return agent

def debug_trained_model():
    """调试已训练模型，检查训练测试差异"""
    print("=== 调试稳定版Gumbel-Softmax PPO模型 ===")
    
    # 创建环境和智能体
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    agent = StableGumbelPPORacetrackAgent(env=env)
    
    # 加载最佳模型
    try:
        agent.load_model("models/stable_gumbel_ppo_model.pth")
        print("✅ 成功加载模型")
    except:
        print("❌ 未找到保存的模型，使用随机初始化")
    
    print(f"当前温度: {agent.temperature}")
    
    # 详细测试几个episode
    print("\n=== 详细调试测试 ===")
    for i in range(3):
        print(f"\n--- Episode {i+1} ---")
        reward, steps, path, success = agent.test_episode(debug=True)
        print(f"Episode {i+1}: 奖励={reward:.1f}, 步数={steps}, 成功={success}")
    
    # 比较不同策略的成功率
    print(f"\n=== 策略对比测试 ===")
    
    # 1. 测试贪婪策略（当前）
    test_results_greedy = []
    for i in range(20):
        reward, steps, path, success = agent.test_episode()
        test_results_greedy.append(success)
    greedy_success_rate = np.mean(test_results_greedy) * 100
    print(f"贪婪策略成功率: {greedy_success_rate:.1f}%")
    
    # 2. 测试带随机性的策略（模拟训练时）
    print(f"测试带随机性策略（模拟训练时）...")
    test_results_random = []
    original_temp = agent.temperature
    agent.temperature = 1.0  # 使用训练时的温度
    
    for i in range(20):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 300
        
        agent.network.eval()
        with torch.no_grad():
            while steps < max_steps:
                # 使用训练时的策略（带Gumbel-Softmax随机性）
                action, _, _ = agent.select_action(state, training=True)
                next_state, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
                
                state = next_state
        
        success = (done and reward == 100)
        test_results_random.append(success)
    
    agent.temperature = original_temp  # 恢复原温度
    random_success_rate = np.mean(test_results_random) * 100
    print(f"随机策略成功率: {random_success_rate:.1f}%")
    
    # 分析结果
    print(f"\n=== 分析结果 ===")
    print(f"贪婪策略（测试）: {greedy_success_rate:.1f}%")
    print(f"随机策略（训练）: {random_success_rate:.1f}%")
    
    if random_success_rate > greedy_success_rate:
        print("🚨 确认问题：随机探索比确定性策略更成功！")
        print("   这说明网络没有学到有效的确定性策略")
        print("   训练时的成功主要来自随机探索，而非策略学习")
    else:
        print("✅ 策略学习正常：确定性策略优于随机策略")
    
    return agent

if __name__ == "__main__":
    # main_stable_gumbel_ppo_training()
    debug_trained_model() 