import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from typing import Dict, List, Tuple
import pickle
import os

from racetrack_env import RacetrackEnv
from sarsa_lambda import SarsaLambdaAgent
from q_learning import QLearningAgent
from reinforce import REINFORCEAgent


def smooth_curve(points: List[float], window_size: int = 100) -> List[float]:
    """平滑曲线"""
    if len(points) < window_size:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed.append(np.mean(points[start:end]))
    
    return smoothed


def evaluate_agent(agent, n_episodes: int = 100) -> Tuple[float, float, float, float]:
    """评估智能体性能"""
    rewards = []
    steps_list = []
    
    for _ in range(n_episodes):
        reward, steps, _ = agent.test_episode()
        rewards.append(reward)
        steps_list.append(steps)
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    
    return avg_reward, std_reward, avg_steps, std_steps


def train_and_compare_algorithms():
    """训练并比较三种算法"""
    print("开始强化学习算法比较实验...")
    
    # 创建环境
    env = RacetrackEnv(track_size=(20, 15), max_speed=3)
    print(f"赛道大小: {env.track_size}")
    print(f"动作空间: {env.n_actions}")
    print(f"起点数量: {len(env.start_positions)}")
    print(f"终点数量: {len(env.goal_positions)}")
    
    # 训练参数
    n_episodes = 2000
    
    # 1. 训练 Sarsa(λ)
    print("\n训练 Sarsa(λ) 智能体...")
    sarsa_agent = SarsaLambdaAgent(
        env=env, 
        alpha=0.1, 
        gamma=0.95, 
        lambda_=0.9, 
        epsilon=0.1
    )
    sarsa_rewards, sarsa_steps = sarsa_agent.train(n_episodes, verbose=True)
    
    # 2. 训练 Q-learning
    print("\n训练 Q-learning 智能体...")
    qlearning_agent = QLearningAgent(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1
    )
    qlearning_rewards, qlearning_steps = qlearning_agent.train(n_episodes, verbose=True)
    
    # 3. 训练 REINFORCE
    print("\n训练 REINFORCE 智能体...")
    reinforce_agent = REINFORCEAgent(
        env=env,
        lr=0.001,
        gamma=0.95,
        hidden_dim=128
    )
    reinforce_rewards, reinforce_steps = reinforce_agent.train(n_episodes, verbose=True)
    
    # 评估最终性能
    print("\n最终性能评估...")
    
    # 设置较小的epsilon用于评估（更贪婪）
    sarsa_agent.epsilon = 0.01
    qlearning_agent.epsilon = 0.01
    
    sarsa_eval = evaluate_agent(sarsa_agent)
    qlearning_eval = evaluate_agent(qlearning_agent)
    reinforce_eval = evaluate_agent(reinforce_agent)
    
    print(f"\nSarsa(λ) - 平均奖励: {sarsa_eval[0]:.2f}±{sarsa_eval[1]:.2f}, 平均步数: {sarsa_eval[2]:.2f}±{sarsa_eval[3]:.2f}")
    print(f"Q-learning - 平均奖励: {qlearning_eval[0]:.2f}±{qlearning_eval[1]:.2f}, 平均步数: {qlearning_eval[2]:.2f}±{qlearning_eval[3]:.2f}")
    print(f"REINFORCE - 平均奖励: {reinforce_eval[0]:.2f}±{reinforce_eval[1]:.2f}, 平均步数: {reinforce_eval[2]:.2f}±{reinforce_eval[3]:.2f}")
    
    # 创建可视化
    create_visualizations(
        sarsa_rewards, sarsa_steps,
        qlearning_rewards, qlearning_steps,
        reinforce_rewards, reinforce_steps,
        sarsa_eval, qlearning_eval, reinforce_eval
    )
    
    # 展示学习到的路径
    print("\n展示学习到的路径...")
    test_and_show_paths(sarsa_agent, qlearning_agent, reinforce_agent, env)
    
    # 保存结果
    results = {
        'sarsa_rewards': sarsa_rewards,
        'sarsa_steps': sarsa_steps,
        'qlearning_rewards': qlearning_rewards,
        'qlearning_steps': qlearning_steps,
        'reinforce_rewards': reinforce_rewards,
        'reinforce_steps': reinforce_steps,
        'evaluations': {
            'sarsa': sarsa_eval,
            'qlearning': qlearning_eval,
            'reinforce': reinforce_eval
        }
    }
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n实验完成！结果已保存到 results.pkl")
    

def create_visualizations(sarsa_rewards, sarsa_steps, 
                         qlearning_rewards, qlearning_steps,
                         reinforce_rewards, reinforce_steps,
                         sarsa_eval, qlearning_eval, reinforce_eval):
    """创建可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练奖励曲线
    ax1 = axes[0, 0]
    episodes = range(len(sarsa_rewards))
    
    ax1.plot(episodes, smooth_curve(sarsa_rewards), label='Sarsa(λ)', alpha=0.8)
    ax1.plot(episodes, smooth_curve(qlearning_rewards), label='Q-learning', alpha=0.8)
    ax1.plot(episodes, smooth_curve(reinforce_rewards), label='REINFORCE', alpha=0.8)
    
    ax1.set_xlabel('训练回合数')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('训练过程中的奖励变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练步数曲线
    ax2 = axes[0, 1]
    
    ax2.plot(episodes, smooth_curve(sarsa_steps), label='Sarsa(λ)', alpha=0.8)
    ax2.plot(episodes, smooth_curve(qlearning_steps), label='Q-learning', alpha=0.8)
    ax2.plot(episodes, smooth_curve(reinforce_steps), label='REINFORCE', alpha=0.8)
    
    ax2.set_xlabel('训练回合数')
    ax2.set_ylabel('平均步数')
    ax2.set_title('训练过程中的步数变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终性能比较 - 奖励
    ax3 = axes[1, 0]
    
    algorithms = ['Sarsa(λ)', 'Q-learning', 'REINFORCE']
    avg_rewards = [sarsa_eval[0], qlearning_eval[0], reinforce_eval[0]]
    std_rewards = [sarsa_eval[1], qlearning_eval[1], reinforce_eval[1]]
    
    bars = ax3.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5)
    ax3.set_ylabel('平均奖励')
    ax3.set_title('最终性能比较 - 平均奖励')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, avg, std) in enumerate(zip(bars, avg_rewards, std_rewards)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{avg:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # 4. 最终性能比较 - 步数
    ax4 = axes[1, 1]
    
    avg_steps = [sarsa_eval[2], qlearning_eval[2], reinforce_eval[2]]
    std_steps = [sarsa_eval[3], qlearning_eval[3], reinforce_eval[3]]
    
    bars = ax4.bar(algorithms, avg_steps, yerr=std_steps, capsize=5)
    ax4.set_ylabel('平均步数')
    ax4.set_title('最终性能比较 - 平均步数')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, avg, std) in enumerate(zip(bars, avg_steps, std_steps)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{avg:.1f}±{std:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("性能比较图表已保存到 algorithm_comparison.png")


def test_and_show_paths(sarsa_agent, qlearning_agent, reinforce_agent, env):
    """测试并展示学习到的路径"""
    
    # 设置绘图参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 为每个智能体生成路径
    agents = [
        ('Sarsa(λ)', sarsa_agent),
        ('Q-learning', qlearning_agent),
        ('REINFORCE', reinforce_agent)
    ]
    
    for idx, (name, agent) in enumerate(agents):
        ax = axes[idx]
        
        # 运行一次测试获取路径
        _, _, path = agent.test_episode(render=False)
        
        # 显示赛道
        display_map = env.track.copy().astype(float)
        
        # 绘制赛道
        im = ax.imshow(display_map, cmap='tab10', vmin=0, vmax=3, alpha=0.8)
        
        # 绘制路径
        if len(path) > 1:
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            ax.plot(path_y, path_x, 'b-', linewidth=3, alpha=0.8, label='学习路径')
            ax.plot(path_y[0], path_x[0], 'go', markersize=10, label='起点')
            ax.plot(path_y[-1], path_x[-1], 'ro', markersize=10, label='终点')
        
        ax.set_title(f'{name} 学习到的路径')
        ax.set_xlabel('Y坐标')
        ax.set_ylabel('X坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learned_paths.png', dpi=300, bbox_inches='tight')
    print("学习路径图已保存到 learned_paths.png")


if __name__ == "__main__":
    train_and_compare_algorithms() 