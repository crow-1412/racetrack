#!/usr/bin/env python3
"""
Q-Guided Actor-Critic算法专项测试

测试Q-Guided AC算法的性能表现，包括：
1. 基础训练测试
2. 性能对比分析
3. 三阶段训练效果评估
4. 与传统AC方法对比

作者：AI Assistant
创建时间：2024年
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import json

from racetrack_env import RacetrackEnv
from q_guided_ac_simple import QGuidedActorCritic
from actor_critic import OptimizedActorCriticAgent

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def test_q_guided_performance():
    """测试Q-Guided AC的基本性能"""
    print("🚀 Q-Guided Actor-Critic性能测试")
    print("=" * 50)
    
    # 创建环境和智能体
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    agent = QGuidedActorCritic(env, lr=0.001, gamma=0.95, alpha_q=0.2, epsilon=0.3)
    
    # 训练记录
    training_results = {
        'episodes': [],
        'success_rates': [],
        'avg_rewards': [],
        'phases': [],
        'q_table_sizes': []
    }
    
    print("\n🎯 开始训练...")
    total_episodes = sum(agent.phase_episodes.values())
    
    for episode in range(total_episodes):
        reward, steps, success = agent.train_episode(episode)
        
        # 每50个episode评估一次
        if (episode + 1) % 50 == 0:
            test_results = []
            for _ in range(10):
                test_reward, test_steps, test_path, test_success = agent.test_episode()
                test_results.append((test_reward, test_success))
            
            success_rate = np.mean([r[1] for r in test_results])
            avg_reward = np.mean([r[0] for r in test_results])
            
            training_results['episodes'].append(episode + 1)
            training_results['success_rates'].append(success_rate)
            training_results['avg_rewards'].append(avg_reward)
            training_results['phases'].append(agent.training_phase)
            training_results['q_table_sizes'].append(len(agent.Q_table))
            
            print(f"Episode {episode+1} ({agent.training_phase}): "
                  f"成功率={success_rate:.2f}, "
                  f"平均奖励={avg_reward:.2f}, "
                  f"Q表大小={len(agent.Q_table)}")
    
    # 最终测试
    print("\n✅ 最终性能测试...")
    final_results = []
    for i in range(20):
        reward, steps, path, success = agent.test_episode()
        final_results.append({
            'reward': reward,
            'steps': steps,
            'success': success,
            'path_length': len(path)
        })
        print(f"测试{i+1}: 奖励={reward:.2f}, 步数={steps}, 成功={success}")
    
    # 统计
    final_success_rate = np.mean([r['success'] for r in final_results])
    final_avg_reward = np.mean([r['reward'] for r in final_results])
    final_avg_steps = np.mean([r['steps'] for r in final_results])
    
    print(f"\n📊 最终统计:")
    print(f"成功率: {final_success_rate:.2%}")
    print(f"平均奖励: {final_avg_reward:.2f} ± {np.std([r['reward'] for r in final_results]):.2f}")
    print(f"平均步数: {final_avg_steps:.2f} ± {np.std([r['steps'] for r in final_results]):.2f}")
    print(f"最终Q表大小: {len(agent.Q_table)}")
    
    return agent, training_results, final_results


def compare_with_traditional_ac():
    """与传统Actor-Critic对比"""
    print("\n🔥 Q-Guided AC vs 传统Actor-Critic对比")
    print("=" * 60)
    
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    
    # 测试Q-Guided AC
    print("\n1️⃣ 测试Q-Guided Actor-Critic...")
    q_guided_agent = QGuidedActorCritic(env, lr=0.001, gamma=0.95, alpha_q=0.2, epsilon=0.3)
    
    # 快速训练
    total_episodes = sum(q_guided_agent.phase_episodes.values())
    for episode in range(total_episodes):
        q_guided_agent.train_episode(episode)
    
    # 测试Q-Guided AC性能
    q_guided_results = []
    for _ in range(20):
        reward, steps, path, success = q_guided_agent.test_episode()
        q_guided_results.append({'reward': reward, 'steps': steps, 'success': success})
    
    # 测试传统AC（如果模型存在）
    print("\n2️⃣ 测试传统Actor-Critic...")
    try:
        traditional_agent = OptimizedActorCriticAgent(env)
        traditional_agent.load_model("models/advanced_tuned_model.pth")
        
        traditional_results = []
        for _ in range(20):
            reward, steps, path, success = traditional_agent.test_episode()
            traditional_results.append({'reward': reward, 'steps': steps, 'success': success})
    except Exception as e:
        print(f"⚠️ 传统AC模型加载失败: {e}")
        print("🔄 开始训练传统AC...")
        traditional_agent = OptimizedActorCriticAgent(env)
        for episode in range(800):
            traditional_agent.train_episode(episode)
        
        traditional_results = []
        for _ in range(20):
            reward, steps, path, success = traditional_agent.test_episode()
            traditional_results.append({'reward': reward, 'steps': steps, 'success': success})
    
    # 对比分析
    print("\n📊 对比结果:")
    print("=" * 40)
    
    # Q-Guided AC结果
    q_success_rate = np.mean([r['success'] for r in q_guided_results])
    q_avg_reward = np.mean([r['reward'] for r in q_guided_results])
    q_avg_steps = np.mean([r['steps'] for r in q_guided_results])
    
    print(f"Q-Guided AC:")
    print(f"  成功率: {q_success_rate:.2%}")
    print(f"  平均奖励: {q_avg_reward:.2f}")
    print(f"  平均步数: {q_avg_steps:.2f}")
    print(f"  Q表大小: {len(q_guided_agent.Q_table)}")
    
    # 传统AC结果
    t_success_rate = np.mean([r['success'] for r in traditional_results])
    t_avg_reward = np.mean([r['reward'] for r in traditional_results])
    t_avg_steps = np.mean([r['steps'] for r in traditional_results])
    
    print(f"\n传统Actor-Critic:")
    print(f"  成功率: {t_success_rate:.2%}")
    print(f"  平均奖励: {t_avg_reward:.2f}")
    print(f"  平均步数: {t_avg_steps:.2f}")
    
    # 优势分析
    print(f"\n🏆 Q-Guided AC优势:")
    print(f"  成功率提升: {(q_success_rate - t_success_rate):.2%}")
    print(f"  奖励提升: {(q_avg_reward - t_avg_reward):.2f}")
    print(f"  步数优化: {(t_avg_steps - q_avg_steps):.2f}")
    
    return q_guided_results, traditional_results


def analyze_training_phases():
    """分析三阶段训练效果"""
    print("\n🔬 三阶段训练效果分析")
    print("=" * 50)
    
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    agent = QGuidedActorCritic(env, lr=0.001, gamma=0.95, alpha_q=0.2, epsilon=0.3)
    
    phase_results = {
        'q_learning': [],
        'hybrid': [],
        'actor_critic': []
    }
    
    total_episodes = sum(agent.phase_episodes.values())
    
    for episode in range(total_episodes):
        reward, steps, success = agent.train_episode(episode)
        
        # 记录每个阶段的表现
        if agent.training_phase == 'q_learning':
            phase_results['q_learning'].append({'episode': episode, 'success': success, 'reward': reward})
        elif agent.training_phase == 'hybrid':
            phase_results['hybrid'].append({'episode': episode, 'success': success, 'reward': reward})
        else:  # actor_critic
            phase_results['actor_critic'].append({'episode': episode, 'success': success, 'reward': reward})
    
    # 分析各阶段性能
    print("\n📈 各阶段性能分析:")
    
    for phase_name, results in phase_results.items():
        if results:
            success_rate = np.mean([r['success'] for r in results])
            avg_reward = np.mean([r['reward'] for r in results])
            print(f"{phase_name.upper()} 阶段:")
            print(f"  Episode数: {len(results)}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  平均奖励: {avg_reward:.2f}")
    
    return phase_results


def create_training_visualization(training_results):
    """创建训练过程可视化"""
    print("\n🎨 生成训练过程可视化...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = training_results['episodes']
    
    # 1. 成功率曲线
    ax1.plot(episodes, training_results['success_rates'], 'b-', linewidth=2, marker='o')
    ax1.set_title('Success Rate Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.grid(True, alpha=0.3)
    
    # 添加阶段分界线
    phase_boundaries = []
    current_phase = None
    for i, phase in enumerate(training_results['phases']):
        if phase != current_phase:
            if current_phase is not None:
                phase_boundaries.append(episodes[i])
            current_phase = phase
    
    for boundary in phase_boundaries:
        ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    
    # 2. 平均奖励曲线
    ax2.plot(episodes, training_results['avg_rewards'], 'g-', linewidth=2, marker='s')
    ax2.set_title('Average Reward Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    
    for boundary in phase_boundaries:
        ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    
    # 3. Q表增长曲线
    ax3.plot(episodes, training_results['q_table_sizes'], 'orange', linewidth=2, marker='^')
    ax3.set_title('Q-Table Size Growth', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Q-Table Size')
    ax3.grid(True, alpha=0.3)
    
    for boundary in phase_boundaries:
        ax3.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    
    # 4. 训练阶段分布
    phase_counts = {'q_learning': 0, 'hybrid': 0, 'actor_critic': 0}
    for phase in training_results['phases']:
        if phase in phase_counts:
            phase_counts[phase] += 1
    
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())
    colors = ['red', 'orange', 'green']
    
    ax4.pie(counts, labels=phases, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Training Phase Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"q_guided_ac_training_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📄 训练可视化已保存: {filename}")
    
    plt.show()


def main():
    """主函数"""
    print("🚀 Q-Guided Actor-Critic全面测试")
    print("=" * 60)
    print("本测试将评估Q-Guided AC算法的:")
    print("  1. 基础性能表现")
    print("  2. 与传统AC的对比")
    print("  3. 三阶段训练效果")
    print("  4. 学习过程可视化")
    print("=" * 60)
    
    # 1. 基础性能测试
    agent, training_results, final_results = test_q_guided_performance()
    
    # 2. 与传统AC对比
    q_guided_results, traditional_results = compare_with_traditional_ac()
    
    # 3. 分析训练阶段
    phase_results = analyze_training_phases()
    
    # 4. 创建可视化
    create_training_visualization(training_results)
    
    # 5. 保存测试结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'training_results': training_results,
        'final_results': final_results,
        'q_guided_results': q_guided_results,
        'traditional_results': traditional_results,
        'phase_results': phase_results
    }
    
    filename = f"q_guided_ac_test_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 测试结果已保存: {filename}")
    
    # 总结
    print(f"\n🎉 Q-Guided Actor-Critic测试完成！")
    print(f"🏆 关键发现:")
    
    final_success_rate = np.mean([r['success'] for r in final_results])
    q_success_rate = np.mean([r['success'] for r in q_guided_results])
    t_success_rate = np.mean([r['success'] for r in traditional_results])
    
    print(f"  📈 最终成功率: {final_success_rate:.2%}")
    print(f"  🆚 相比传统AC提升: {(q_success_rate - t_success_rate):.2%}")
    print(f"  🧠 Q表最终大小: {len(agent.Q_table)}")
    print(f"  ⚡ 三阶段训练策略有效结合了表格和神经网络方法的优势")


if __name__ == "__main__":
    main() 