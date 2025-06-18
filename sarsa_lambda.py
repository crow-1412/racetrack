import numpy as np
from typing import Tuple, List, Dict, Any
import random
from racetrack_env import RacetrackEnv


class SarsaLambdaAgent:
    """
    Sarsa(λ) 智能体
    使用资格迹加速学习
    """
    
    def __init__(self, env: RacetrackEnv, alpha=0.1, gamma=0.95, lambda_=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        
        # 初始化Q表和资格迹
        self.state_space_size = env.get_state_space_size()
        self.n_actions = env.n_actions
        
        # 使用字典存储Q值和资格迹（稀疏表示）
        self.Q: Dict[Tuple[Tuple[int, int, int, int], int], float] = {}
        self.E: Dict[Tuple[Tuple[int, int, int, int], int], float] = {}  # 资格迹
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
    
    def get_q_value(self, state: Tuple[int, int, int, int], action: int) -> float:
        """获取Q值"""
        state_key = state
        if (state_key, action) not in self.Q:
            self.Q[(state_key, action)] = 0.0
        return self.Q[(state_key, action)]
    
    def set_q_value(self, state: Tuple[int, int, int, int], action: int, value: float):
        """设置Q值"""
        state_key = state
        self.Q[(state_key, action)] = value
    
    def get_eligibility(self, state: Tuple[int, int, int, int], action: int) -> float:
        """获取资格迹值"""
        state_key = state
        if (state_key, action) not in self.E:
            self.E[(state_key, action)] = 0.0
        return self.E[(state_key, action)]
    
    def set_eligibility(self, state: Tuple[int, int, int, int], action: int, value: float):
        """设置资格迹值"""
        state_key = state
        if value == 0.0:
            # 删除零值以节省内存
            if (state_key, action) in self.E:
                del self.E[(state_key, action)]
        else:
            self.E[(state_key, action)] = value
    
    def epsilon_greedy_action(self, state: Tuple[int, int, int, int]) -> int:
        """ε-greedy 动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # 贪婪选择
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            max_q = max(q_values)
            # 如果有多个最大值，随机选择一个
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个episode"""
        state = self.env.reset()
        action = self.epsilon_greedy_action(state)
        
        # 清零资格迹
        self.E.clear()
        
        total_reward = 0.0
        steps = 0
        
        while True:
            # 执行动作
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # 选择下一个动作
            next_action = self.epsilon_greedy_action(next_state)
            
            # 计算TD误差
            q_current = self.get_q_value(state, action)
            q_next = self.get_q_value(next_state, next_action) if not done else 0.0
            delta = reward + self.gamma * q_next - q_current
            
            # 更新当前状态-动作的资格迹
            current_eligibility = self.get_eligibility(state, action)
            self.set_eligibility(state, action, current_eligibility + 1.0)
            
            # 更新所有状态-动作对的Q值和资格迹
            states_actions_to_update = list(self.E.keys())
            for (s, a) in states_actions_to_update:
                eligibility = self.get_eligibility(s, a)
                if eligibility > 1e-10:  # 只更新非零资格迹
                    # 更新Q值
                    old_q = self.get_q_value(s, a)
                    new_q = old_q + self.alpha * delta * eligibility
                    self.set_q_value(s, a, new_q)
                    
                    # 衰减资格迹
                    new_eligibility = self.gamma * self.lambda_ * eligibility
                    self.set_eligibility(s, a, new_eligibility)
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        return total_reward, steps
    
    def train(self, n_episodes: int, verbose: bool = True) -> Tuple[List[float], List[int]]:
        """训练智能体"""
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
        """测试一个episode（默认使用贪婪策略）"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state[:2]]  # 记录路径（只记录位置）
        
        while steps < 1000:  # 防止无限循环
            if use_exploration:
                # 使用ε-greedy策略
                action = self.epsilon_greedy_action(state)
            else:
                # 贪婪选择动作
                q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
                action = int(np.argmax(q_values))
            
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
    
    def get_policy(self) -> Dict[Tuple[int, int, int, int], int]:
        """获取当前策略（贪婪策略）"""
        policy: Dict[Tuple[int, int, int, int], int] = {}
        for (state, action), q_value in self.Q.items():
            if state not in policy:
                policy[state] = action
            else:
                current_q = self.get_q_value(state, policy[state])
                if q_value > current_q:
                    policy[state] = action
        return policy


def main():
    """
    主函数：演示 Sarsa(λ) 智能体的训练和测试过程
    """
    print("=== Sarsa(λ) 智能体训练演示 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 最大速度: {env.max_speed}")
    print(f"  - 动作数量: {env.n_actions}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    
    # 创建智能体 - 调整参数以适应新环境
    agent = SarsaLambdaAgent(
        env=env,
        alpha=0.2,      # 增加学习率
        gamma=0.95,     # 折扣因子
        lambda_=0.8,    # 略微降低资格迹衰减系数
        epsilon=0.2     # 增加探索率
    )
    print(f"\n智能体参数：")
    print(f"  - 学习率 α: {agent.alpha}")
    print(f"  - 折扣因子 γ: {agent.gamma}")
    print(f"  - 资格迹系数 λ: {agent.lambda_}")
    print(f"  - 探索率 ε: {agent.epsilon}")
    
    # 训练前测试
    print("\n=== 训练前测试 ===")
    reward_before, steps_before, path_before = agent.test_episode()
    print(f"训练前性能: 奖励 = {reward_before:.2f}, 步数 = {steps_before}")
    print(f"是否到达终点: {'是' if path_before[-1] in env.goal_positions else '否'}")
    
    # 训练智能体 - 增加训练轮数
    print("\n=== 开始训练 ===")
    n_episodes = 2000  # 增加训练轮数
    rewards, steps = agent.train(n_episodes=n_episodes, verbose=True)
    
    # 分析训练结果
    print(f"\n=== 训练结果分析 ===")
    print(f"总训练回合数: {n_episodes}")
    print(f"Q表大小: {len(agent.Q)} 个状态-动作对")
    print(f"最终100回合平均奖励: {np.mean(rewards[-100:]):.2f}")
    print(f"最终100回合平均步数: {np.mean(steps[-100:]):.2f}")
    
    # 训练后测试
    print("\n=== 训练后测试 ===")
    test_episodes = 10
    test_rewards = []
    test_steps = []
    test_paths = []
    success_count = 0
    
    for i in range(test_episodes):
        reward, step_count, path = agent.test_episode()
        test_rewards.append(reward)
        test_steps.append(step_count)
        test_paths.append(path)
        is_success = path[-1] in env.goal_positions
        if is_success:
            success_count += 1
        print(f"测试 {i+1}: 奖励 = {reward:.2f}, 步数 = {step_count}, "
              f"成功 = {'是' if is_success else '否'}")
    
    print(f"\n=== 最终性能统计 ===")
    print(f"测试回合数: {test_episodes}")
    print(f"成功率: {success_count}/{test_episodes} = {success_count/test_episodes*100:.1f}%")
    print(f"平均奖励: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"平均步数: {np.mean(test_steps):.2f} ± {np.std(test_steps):.2f}")
    
    # 从测试结果中找到最优路径
    print(f"\n=== 展示最优路径 ===")
    # 找到奖励最高的成功路径
    best_idx = -1
    best_reward = float('-inf')
    for i, (reward, path) in enumerate(zip(test_rewards, test_paths)):
        if path[-1] in env.goal_positions and reward > best_reward:
            best_reward = reward
            best_idx = i
    
    if best_idx >= 0:
        best_reward = test_rewards[best_idx]
        best_steps = test_steps[best_idx]
        best_path = test_paths[best_idx]
        print(f"最优路径 (来自测试 {best_idx+1}): 奖励 = {best_reward:.2f}, 步数 = {best_steps}")
        print(f"路径长度: {len(best_path)} 个位置")
    else:
        print("未找到成功的路径，显示奖励最高的路径:")
        best_idx = np.argmax(test_rewards)
        best_reward = test_rewards[best_idx]
        best_steps = test_steps[best_idx]
        best_path = test_paths[best_idx]
        print(f"最佳尝试 (来自测试 {best_idx+1}): 奖励 = {best_reward:.2f}, 步数 = {best_steps}")
        print(f"路径长度: {len(best_path)} 个位置")
    
    if len(best_path) <= 20:  # 如果路径不太长，显示完整路径
        print("完整路径:", " -> ".join([f"({x},{y})" for x, y in best_path]))
    else:
        print("起始路径:", " -> ".join([f"({x},{y})" for x, y in best_path[:5]]))
        print("......")
        print("结束路径:", " -> ".join([f"({x},{y})" for x, y in best_path[-5:]]))
    
    # 策略分析
    policy = agent.get_policy()
    print(f"\n=== 策略分析 ===")
    print(f"学到的策略包含 {len(policy)} 个状态")
    
    # 显示起点附近的策略
    print("起点附近的策略示例:")
    for state in list(policy.keys())[:5]:
        action_idx = policy[state]
        action = env.actions[action_idx]
        print(f"  状态 {state} -> 动作 {action}")
    
    # 显示训练进度
    if len(rewards) > 100:
        print(f"\n=== 学习进度 ===")
        # 计算每100回合的平均性能
        progress_points = []
        for i in range(100, len(rewards) + 1, 100):
            avg_reward = np.mean(rewards[i-100:i])
            avg_steps = np.mean(steps[i-100:i])
            progress_points.append((i, avg_reward, avg_steps))
            print(f"回合 {i}: 平均奖励 = {avg_reward:.2f}, 平均步数 = {avg_steps:.2f}")
    
    return agent, rewards, steps


if __name__ == "__main__":
    main() 