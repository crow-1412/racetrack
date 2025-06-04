#!/usr/bin/env python3
"""
简单测试脚本，验证环境和算法是否正常工作
"""

import numpy as np
from racetrack_env import RacetrackEnv
from sarsa_lambda import SarsaLambdaAgent
from q_learning import QLearningAgent
from reinforce import REINFORCEAgent

def test_environment():
    """测试环境基本功能"""
    print("测试环境...")
    
    env = RacetrackEnv(track_size=(10, 8), max_speed=2)
    print(f"环境创建成功: {env.track_size}")
    print(f"动作空间大小: {env.n_actions}")
    print(f"起点数量: {len(env.start_positions)}")
    print(f"终点数量: {len(env.goal_positions)}")
    
    # 测试重置
    state = env.reset()
    print(f"初始状态: {state}")
    
    # 测试几步动作
    for i in range(5):
        action = np.random.randint(0, env.n_actions)
        next_state, reward, done = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 状态={next_state}, 奖励={reward}, 完成={done}")
        if done:
            break
    
    print("环境测试完成！\n")

def test_sarsa():
    """测试Sarsa(λ)算法"""
    print("测试 Sarsa(λ) 算法...")
    
    env = RacetrackEnv(track_size=(10, 8), max_speed=2)
    agent = SarsaLambdaAgent(env, alpha=0.1, gamma=0.95, lambda_=0.9, epsilon=0.1)
    
    # 训练几个episode
    print("训练 10 个回合...")
    rewards, steps = agent.train(10, verbose=False)
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"平均步数: {np.mean(steps):.2f}")
    
    # 测试一个episode
    reward, steps, path = agent.test_episode()
    print(f"测试结果: 奖励={reward}, 步数={steps}, 路径长度={len(path)}")
    
    print("Sarsa(λ) 测试完成！\n")

def test_qlearning():
    """测试Q-learning算法"""
    print("测试 Q-learning 算法...")
    
    env = RacetrackEnv(track_size=(10, 8), max_speed=2)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    # 训练几个episode
    print("训练 10 个回合...")
    rewards, steps = agent.train(10, verbose=False)
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"平均步数: {np.mean(steps):.2f}")
    
    # 测试一个episode
    reward, steps, path = agent.test_episode()
    print(f"测试结果: 奖励={reward}, 步数={steps}, 路径长度={len(path)}")
    
    print("Q-learning 测试完成！\n")

def test_reinforce():
    """测试REINFORCE算法"""
    print("测试 REINFORCE 算法...")
    
    env = RacetrackEnv(track_size=(10, 8), max_speed=2)
    agent = REINFORCEAgent(env, lr=0.01, gamma=0.95, hidden_dim=64)
    
    # 训练几个episode
    print("训练 10 个回合...")
    rewards, steps = agent.train(10, verbose=False)
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"平均步数: {np.mean(steps):.2f}")
    
    # 测试一个episode
    reward, steps, path = agent.test_episode()
    print(f"测试结果: 奖励={reward}, 步数={steps}, 路径长度={len(path)}")
    
    print("REINFORCE 测试完成！\n")

if __name__ == "__main__":
    print("开始运行测试...\n")
    
    try:
        test_environment()
        test_sarsa()
        test_qlearning()
        test_reinforce()
        print("所有测试通过！代码运行正常。")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc() 