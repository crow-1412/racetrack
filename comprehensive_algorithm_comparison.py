#!/usr/bin/env python3
"""
强化学习算法全面对比脚本

对比以下算法：
1. REINFORCE (Policy Gradient)
2. Actor-Critic  
3. PPO (Proximal Policy Optimization)
4. TRPO (Trust Region Policy Optimization)
5. Q-Learning (Value-based)
6. Sarsa(λ) (Value-based)

对比维度：
- 收敛速度 (训练曲线)
- 平均步数 (执行效率)
- 策略稳定性 (多次测试方差)
- 成功率 (任务完成率)
- 学习效率 (样本利用率)

控制变量设计：
- 所有算法使用相同的环境参数
- 测试时使用固定随机种子确保可重现性
- 相同的测试次数和稳定性测试批次
- 统一的性能评估标准

作者： YuJinYue
创建时间：2025年6月19日
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

        # 导入所有算法
from racetrack_env import RacetrackEnv
from reinforce import OptimizedREINFORCEAgent, test_saved_reinforce_model
from actor_critic import OptimizedActorCriticAgent  
from ppo import OptimizedPPORacetrackAgent
from trpo_racetrack import TRPORacetrackAgent
from q_learning import QLearningAgent
from sarsa_lambda import SarsaLambdaAgent
from q_guided_ac_simple import QGuidedActorCritic, UltraOptimizedQGAC

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AlgorithmComparator:
    """强化学习算法全面对比器"""
    
    def __init__(self, track_size=(32, 17), max_speed=5):
        self.env = RacetrackEnv(track_size=track_size, max_speed=max_speed)
        self.results = {}
        self.algorithms = {}
        self.trained_models = {}
        
        # 对比配置
        self.test_episodes = 100  # 测试episode数
        self.training_episodes = 1000  # 快速训练episode数
        self.stability_tests = 20  # 稳定性测试次数
        
        # 控制变量：确保所有算法使用相同的测试条件
        self.controlled_test_seeds = list(range(1000, 1000 + self.test_episodes))  # 固定种子
        self.controlled_stability_seeds = list(range(2000, 2000 + self.stability_tests * 10))  # 稳定性测试种子
        
        print("🚀 强化学习算法全面对比器已初始化")
        print(f"📍 环境配置: {track_size}, 最大速度: {max_speed}")
    
    def load_or_train_all_algorithms(self):
        """加载或训练所有算法"""
        print("\n" + "="*60)
        print("🔄 加载或训练所有算法")
        print("="*60)
        
        # 1. REINFORCE
        print("\n1️⃣ 处理REINFORCE算法...")
        try:
            self.algorithms['REINFORCE'] = OptimizedREINFORCEAgent(self.env)
            self.algorithms['REINFORCE'].load_model("models/optimized_reinforce_model.pth")
            print("✅ REINFORCE模型加载成功")
        except Exception as e:
            print(f"⚠️ REINFORCE模型加载失败: {e}")
            print("🔄 开始快速训练REINFORCE...")
            self.algorithms['REINFORCE'] = self._quick_train_reinforce()
        
        # 2. Actor-Critic
        print("\n2️⃣ 处理Actor-Critic算法...")
        try:
            self.algorithms['Actor-Critic'] = OptimizedActorCriticAgent(self.env)
            self.algorithms['Actor-Critic'].load_model("models/advanced_tuned_model.pth")
            print("✅ Actor-Critic模型加载成功")
        except Exception as e:
            print(f"⚠️ Actor-Critic模型加载失败: {e}")
            print("🔄 开始快速训练Actor-Critic...")
            self.algorithms['Actor-Critic'] = self._quick_train_actor_critic()
        
        # 3. PPO
        print("\n3️⃣ 处理PPO算法...")
        try:
            self.algorithms['PPO'] = OptimizedPPORacetrackAgent(self.env)
            self.algorithms['PPO'].load_model("models/optimized_ppo_racetrack_model.pth")
            print("✅ PPO模型加载成功")
        except Exception as e:
            print(f"⚠️ PPO模型加载失败: {e}")
            print("🔄 开始快速训练PPO...")
            self.algorithms['PPO'] = self._quick_train_ppo()
        
        # 4. TRPO
        print("\n4️⃣ 处理TRPO算法...")
        try:
            self.algorithms['TRPO'] = TRPORacetrackAgent(self.env)
            self.algorithms['TRPO'].load_model("models/trpo_racetrack_model.pth")
            print("✅ TRPO模型加载成功")
        except Exception as e:
            print(f"⚠️ TRPO模型加载失败: {e}")
            print("🔄 开始快速训练TRPO...")
            self.algorithms['TRPO'] = self._quick_train_trpo()
        
        # 5. Q-Learning
        print("\n5️⃣ 处理Q-Learning算法...")
        print("🔄 Q-Learning需要重新训练（基于值函数）...")
        self.algorithms['Q-Learning'] = self._quick_train_qlearning()
        
        # 6. Sarsa(λ)
        print("\n6️⃣ 处理Sarsa(λ)算法...")
        print("🔄 Sarsa(λ)需要重新训练（基于值函数）...")
        self.algorithms['Sarsa(λ)'] = self._quick_train_sarsa_lambda()
        
        # 7. Q-Guided Actor-Critic
        print("\n7️⃣ 处理Q-Guided Actor-Critic算法...")
        print("🔄 Q-Guided AC需要重新训练（混合方法）...")
        self.algorithms['Q-Guided AC'] = self._quick_train_q_guided_ac()
        
        # 8. Ultra-Optimized Q-Guided Actor-Critic
        print("\n8️⃣ 处理超级优化Q-Guided Actor-Critic算法...")
        print("🔄 Ultra Q-Guided AC需要重新训练（超级优化版）...")
        self.algorithms['Ultra Q-Guided AC'] = self._quick_train_ultra_q_guided_ac()
        
        print(f"\n✅ 所有算法准备完成！共{len(self.algorithms)}个算法")
    
    def _quick_train_reinforce(self) -> OptimizedREINFORCEAgent:
        """快速训练REINFORCE"""
        agent = OptimizedREINFORCEAgent(self.env)
        
        print(f"开始{self.training_episodes}轮快速训练...")
        for episode in range(self.training_episodes):
            agent.train_episode(episode)
            
            if (episode + 1) % 200 == 0:
                success_rate = np.mean([1 if r > 0 else 0 for r in agent.episode_rewards[-50:]])
                print(f"Episode {episode+1}: 成功率 ≈ {success_rate:.2f}")
        
        agent.save_model("models/quick_reinforce_model.pth")
        return agent
    
    def _quick_train_actor_critic(self) -> OptimizedActorCriticAgent:
        """快速训练Actor-Critic"""
        agent = OptimizedActorCriticAgent(self.env)
        
        print(f"开始{self.training_episodes}轮快速训练...")
        for episode in range(self.training_episodes):
            agent.train_episode(episode)
            
            if (episode + 1) % 200 == 0:
                success_rate = np.mean([1 if r > 0 else 0 for r in agent.episode_rewards[-50:]])
                print(f"Episode {episode+1}: 成功率 ≈ {success_rate:.2f}")
        
        agent.save_model("models/quick_actor_critic_model.pth")
        return agent
    
    def _quick_train_ppo(self) -> OptimizedPPORacetrackAgent:
        """快速训练PPO"""
        agent = OptimizedPPORacetrackAgent(self.env)
        
        print(f"开始{self.training_episodes}轮快速训练...")
        for episode in range(self.training_episodes):
            agent.train_episode(episode)
            
            if (episode + 1) % 200 == 0:
                success_rate = np.mean([1 if r > 0 else 0 for r in agent.episode_rewards[-50:]])
                print(f"Episode {episode+1}: 成功率 ≈ {success_rate:.2f}")
        
        agent.save_model("models/quick_ppo_model.pth")
        return agent
    
    def _quick_train_trpo(self) -> TRPORacetrackAgent:
        """快速训练TRPO"""
        agent = TRPORacetrackAgent(self.env)
        
        print(f"开始{self.training_episodes}轮快速训练...")
        for episode in range(self.training_episodes):
            agent.train_episode(episode)
            
            if (episode + 1) % 200 == 0:
                success_rate = np.mean([1 if s else 0 for _, _, s in 
                                     [agent.test_episode() for _ in range(10)]])
                print(f"Episode {episode+1}: 成功率 ≈ {success_rate:.2f}")
        
        agent.save_model("models/quick_trpo_model.pth")
        return agent
    
    def _quick_train_qlearning(self) -> QLearningAgent:
        """快速训练Q-Learning"""
        agent = QLearningAgent(self.env, alpha=0.2, gamma=0.95, epsilon=0.15)
        
        print(f"开始{self.training_episodes*2}轮快速训练（Q-Learning需要更多样本）...")
        rewards, steps = agent.train(n_episodes=self.training_episodes*2, verbose=False)
        
        # 显示训练进度
        for i in range(200, len(rewards), 400):
            avg_reward = np.mean(rewards[i-200:i])
            print(f"Episode {i}: 平均奖励 = {avg_reward:.2f}")
        
        return agent
    
    def _quick_train_sarsa_lambda(self) -> SarsaLambdaAgent:
        """快速训练Sarsa(λ)"""
        agent = SarsaLambdaAgent(self.env, alpha=0.15, gamma=0.95, lambda_=0.9, epsilon=0.1)
        
        print(f"开始{self.training_episodes*2}轮快速训练（Sarsa需要更多样本）...")
        rewards, steps = agent.train(n_episodes=self.training_episodes*2, verbose=False)
        
        # 显示训练进度
        for i in range(200, len(rewards), 400):
            avg_reward = np.mean(rewards[i-200:i])
            print(f"Episode {i}: 平均奖励 = {avg_reward:.2f}")
        
        return agent
    
    def _quick_train_q_guided_ac(self) -> QGuidedActorCritic:
        """快速训练Q-Guided Actor-Critic"""
        agent = QGuidedActorCritic(self.env, lr=0.001, gamma=0.95, alpha_q=0.2, epsilon=0.3)
        
        total_episodes = sum(agent.phase_episodes.values())
        print(f"开始{total_episodes}轮快速训练（三阶段）...")
        
        for episode in range(total_episodes):
            reward, steps, success = agent.train_episode(episode)
            
            # 每200个episode报告
            if (episode + 1) % 200 == 0:
                success_rate = np.mean([1 if agent.test_episode()[3] else 0 for _ in range(10)])
                print(f"Episode {episode+1} ({agent.training_phase}): "
                      f"成功率 ≈ {success_rate:.2f}, Q表大小={len(agent.Q_table)}")
        
        return agent
    
    def _quick_train_ultra_q_guided_ac(self) -> UltraOptimizedQGAC:
        """快速训练Ultra-Optimized Q-Guided Actor-Critic"""
        # 使用预分析的最优起点
        best_starts = [(31, 10), (31, 13), (31, 16), (31, 3), (31, 6)]
        agent = UltraOptimizedQGAC(self.env, best_starts=best_starts)
        
        total_episodes = sum(agent.phase_episodes.values())
        print(f"开始{total_episodes}轮快速训练（超级优化三阶段）...")
        
        # 记录训练进度
        training_results = []
        
        for episode in range(total_episodes):
            reward, steps, success = agent.train_episode(episode)
            
            if success:
                training_results.append(steps)
            
            # 每200个episode报告
            if (episode + 1) % 200 == 0:
                recent_successes = [s for s in training_results[-50:]] if len(training_results) >= 50 else training_results
                if recent_successes:
                    avg_recent = np.mean(recent_successes)
                    min_recent = min(recent_successes)
                    print(f"Episode {episode+1} ({agent.training_phase}): "
                          f"成功率 ≈ {len(recent_successes)/50:.2f}, "
                          f"平均步数={avg_recent:.1f}, 最佳={min_recent}步, "
                          f"Q表大小={len(agent.Q_table)}")
                else:
                    print(f"Episode {episode+1} ({agent.training_phase}): "
                          f"Q表大小={len(agent.Q_table)}, ε={agent.epsilon:.4f}")
        
        print(f"超级优化训练完成，最终Q表大小: {len(agent.Q_table)}")
        return agent
    
    def comprehensive_performance_test(self):
        """全面性能测试"""
        print("\n" + "="*60)
        print("🧪 开始全面性能测试")
        print("="*60)
        
        all_results = {}
        
        for alg_name, algorithm in self.algorithms.items():
            print(f"\n🔍 测试 {alg_name}...")
            
            # 基础性能测试
            basic_results = self._test_basic_performance(algorithm, alg_name)
            
            # 稳定性测试
            stability_results = self._test_stability(algorithm, alg_name)
            
            # 合并结果
            all_results[alg_name] = {
                **basic_results,
                **stability_results
            }
            
            print(f"✅ {alg_name} 测试完成")
        
        self.results = all_results
        return all_results
    
    def _test_basic_performance(self, algorithm, alg_name: str) -> Dict:
        """基础性能测试"""
        results = {
            'rewards': [],
            'steps': [],
            'successes': [],
            'paths': [],
            'test_times': []
        }
        
        print(f"  进行{self.test_episodes}次基础测试...")
        
        for i in range(self.test_episodes):
            start_time = time.time()
            
            # 控制变量：为每次测试设置固定随机种子
            test_seed = self.controlled_test_seeds[i]
            np.random.seed(test_seed)
            torch.manual_seed(test_seed)
            
            try:
                # 根据算法类型调用不同的测试方法
                if alg_name in ['Q-Learning', 'Sarsa(λ)']:
                    # Q-Learning 和 Sarsa 类型返回3个值
                    reward, steps, path = algorithm.test_episode()
                    # 改进的成功判断：检查最终位置是否在目标区域
                    success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                else:
                    # 策略方法和Q-Guided AC返回4个值
                    test_result = algorithm.test_episode()
                    if len(test_result) == 4:
                        reward, steps, path, success_from_alg = test_result
                        # 双重验证：算法判断和位置验证
                        position_success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                        success = success_from_alg or position_success
                    else:
                        # 兼容性处理：如果返回3个值
                        reward, steps, path = test_result
                        success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                
                test_time = time.time() - start_time
                
                results['rewards'].append(reward)
                results['steps'].append(steps)
                results['successes'].append(success)
                results['paths'].append(path)
                results['test_times'].append(test_time)
                
            except Exception as e:
                print(f"    ⚠️ 第{i+1}次测试失败: {e}")
                # 记录失败的测试
                results['rewards'].append(-300)
                results['steps'].append(300)
                results['successes'].append(False)
                results['paths'].append([])
                test_time = time.time() - start_time
                results['test_times'].append(test_time)
            
            # 显示进度
            if (i + 1) % 20 == 0:
                current_success_rate = np.mean(results['successes'])
                print(f"    进度: {i+1}/{self.test_episodes}, 当前成功率: {current_success_rate:.2%}")
        
        # 计算统计指标
        success_rate = np.mean(results['successes'])
        avg_reward = np.mean(results['rewards'])
        avg_steps = np.mean(results['steps'])
        avg_test_time = np.mean(results['test_times'])
        
        print(f"  📊 {alg_name} 基础性能:")
        print(f"    成功率: {success_rate:.2%}")
        print(f"    平均奖励: {avg_reward:.2f}")
        print(f"    平均步数: {avg_steps:.1f}")
        print(f"    平均测试时间: {avg_test_time*1000:.2f}ms")
        
        return results
    
    def _test_stability(self, algorithm, alg_name: str) -> Dict:
        """稳定性测试"""
        print(f"  进行{self.stability_tests}次稳定性测试...")
        
        stability_results = []
        
        for i in range(self.stability_tests):
            batch_results = []
            
            # 每次稳定性测试进行10次episode
            for j in range(10):
                # 控制变量：为每次稳定性测试设置固定随机种子
                stability_seed = self.controlled_stability_seeds[i * 10 + j]
                np.random.seed(stability_seed)
                torch.manual_seed(stability_seed)
                
                try:
                    if alg_name in ['Q-Learning', 'Sarsa(λ)']:
                        # Q-Learning 和 Sarsa 类型返回3个值
                        reward, steps, path = algorithm.test_episode()
                        success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                    else:
                        # 策略方法和Q-Guided AC返回4个值
                        test_result = algorithm.test_episode()
                        if len(test_result) == 4:
                            reward, steps, path, success_from_alg = test_result
                            # 双重验证：算法判断和位置验证
                            position_success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                            success = success_from_alg or position_success
                        else:
                            # 兼容性处理：如果返回3个值
                            reward, steps, path = test_result
                            success = (len(path) > 0 and path[-1] in self.env.goal_positions)
                    
                    batch_results.append({
                        'reward': reward,
                        'steps': steps,
                        'success': success
                    })
                except Exception as e:
                    print(f"    ⚠️ 稳定性测试失败: {e}")
                    batch_results.append({
                        'reward': -300,
                        'steps': 300,
                        'success': False
                    })
            
            # 计算此批次的统计量
            batch_success_rate = np.mean([r['success'] for r in batch_results])
            batch_avg_reward = np.mean([r['reward'] for r in batch_results])
            batch_avg_steps = np.mean([r['steps'] for r in batch_results])
            
            stability_results.append({
                'success_rate': batch_success_rate,
                'avg_reward': batch_avg_reward,
                'avg_steps': batch_avg_steps
            })
        
        # 计算稳定性指标
        success_rates = [r['success_rate'] for r in stability_results]
        avg_rewards = [r['avg_reward'] for r in stability_results]
        avg_steps = [r['avg_steps'] for r in stability_results]
        
        stability_metrics = {
            'success_rate_std': np.std(success_rates),
            'reward_std': np.std(avg_rewards),
            'steps_std': np.std(avg_steps),
            'success_rate_var': np.var(success_rates),
            'reward_var': np.var(avg_rewards),
            'steps_var': np.var(avg_steps),
            'stability_batches': stability_results
        }
        
        print(f"  📈 {alg_name} 稳定性:")
        print(f"    成功率标准差: {stability_metrics['success_rate_std']:.4f}")
        print(f"    奖励标准差: {stability_metrics['reward_std']:.2f}")
        print(f"    步数标准差: {stability_metrics['steps_std']:.2f}")
        
        return stability_metrics
    
    def generate_comprehensive_report(self):
        """生成全面对比报告"""
        print("\n" + "="*60)
        print("📊 生成全面对比报告")
        print("="*60)
        
        # 创建对比表格
        comparison_data = []
        
        for alg_name, results in self.results.items():
            success_rate = np.mean(results['successes'])
            avg_reward = np.mean(results['rewards'])
            avg_steps = np.mean(results['steps'])
            reward_std = np.std(results['rewards'])
            steps_std = np.std(results['steps'])
            
            comparison_data.append({
                '算法': alg_name,
                '成功率': f"{success_rate:.2%}",
                '平均奖励': f"{avg_reward:.2f}",
                '平均步数': f"{avg_steps:.1f}",
                '奖励标准差': f"{reward_std:.2f}",
                '步数标准差': f"{steps_std:.2f}",
                '稳定性(成功率方差)': f"{results['success_rate_var']:.6f}",
                '样本效率': f"{success_rate/avg_steps*1000:.2f}" if avg_steps > 0 else "0.00"
            })
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        print("\n🏆 算法性能对比表:")
        print("=" * 100)
        print(df.to_string(index=False))
        
        # 排名分析
        print(f"\n🥇 各项指标排名:")
        print("=" * 50)
        
        # 成功率排名
        success_ranking = sorted(self.results.items(), 
                               key=lambda x: np.mean(x[1]['successes']), reverse=True)
        print("📈 成功率排名:")
        for i, (alg, results) in enumerate(success_ranking):
            success_rate = np.mean(results['successes'])
            print(f"  {i+1}. {alg}: {success_rate:.2%}")
        
        # 平均步数排名（越少越好）
        steps_ranking = sorted([(alg, np.mean(results['steps'])) for alg, results in self.results.items()], 
                             key=lambda x: x[1])
        print(f"\n⚡ 执行效率排名（平均步数，越少越好）:")
        for i, (alg, avg_steps) in enumerate(steps_ranking):
            print(f"  {i+1}. {alg}: {avg_steps:.1f}步")
        
        # 稳定性排名（方差越小越好）
        stability_ranking = sorted([(alg, results['success_rate_var']) for alg, results in self.results.items()], 
                                 key=lambda x: x[1])
        print(f"\n🎯 稳定性排名（成功率方差，越小越好）:")
        for i, (alg, variance) in enumerate(stability_ranking):
            print(f"  {i+1}. {alg}: {variance:.6f}")
        
        # 综合评分
        print(f"\n🏅 综合评分（成功率×0.4 + 效率得分×0.3 + 稳定性得分×0.3）:")
        composite_scores = self._calculate_composite_scores()
        for i, (alg, score) in enumerate(composite_scores):
            print(f"  {i+1}. {alg}: {score:.3f}")
        
        return df, comparison_data
    
    def _calculate_composite_scores(self) -> List[Tuple[str, float]]:
        """计算综合评分"""
        scores = {}
        
        # 获取所有指标的值
        success_rates = [(alg, np.mean(results['successes'])) for alg, results in self.results.items()]
        avg_steps = [(alg, np.mean(results['steps'])) for alg, results in self.results.items()]
        variances = [(alg, results['success_rate_var']) for alg, results in self.results.items()]
        
        # 归一化分数 (0-1)
        max_success = max([s[1] for s in success_rates])
        min_steps = min([s[1] for s in avg_steps])
        max_steps = max([s[1] for s in avg_steps])
        min_var = min([v[1] for v in variances])
        max_var = max([v[1] for v in variances])
        
        for alg in self.results.keys():
            # 成功率得分（越高越好）
            success_score = next(s[1] for s in success_rates if s[0] == alg) / max_success if max_success > 0 else 0
            
            # 效率得分（步数越少越好）
            steps_val = next(s[1] for s in avg_steps if s[0] == alg)
            efficiency_score = (max_steps - steps_val) / (max_steps - min_steps) if max_steps > min_steps else 0
            
            # 稳定性得分（方差越小越好）
            var_val = next(v[1] for v in variances if v[0] == alg)
            stability_score = (max_var - var_val) / (max_var - min_var) if max_var > min_var else 0
            
            # 综合得分
            composite_score = success_score * 0.4 + efficiency_score * 0.3 + stability_score * 0.3
            scores[alg] = composite_score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def create_comprehensive_visualizations(self):
        """创建全面的可视化图表"""
        print("\n🎨 生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 成功率对比柱状图
        ax1 = plt.subplot(2, 3, 1)
        algs = list(self.results.keys())
        success_rates = [np.mean(self.results[alg]['successes']) for alg in algs]
        
        bars = ax1.bar(algs, success_rates)
        ax1.set_title('🏆 Success Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{success_rates[i]:.2%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 2. 平均步数对比
        ax2 = plt.subplot(2, 3, 2)
        avg_steps = [np.mean(self.results[alg]['steps']) for alg in algs]
        
        bars = ax2.bar(algs, avg_steps, color='orange')
        ax2.set_title('⚡ Average Steps Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Steps')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_steps)*0.01,
                    f'{avg_steps[i]:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 3. 奖励分布箱线图
        ax3 = plt.subplot(2, 3, 3)
        reward_data = [self.results[alg]['rewards'] for alg in algs]
        
        box_plot = ax3.boxplot(reward_data, labels=algs)
        ax3.set_title('💰 Reward Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Reward')
        plt.xticks(rotation=45)
        
        # 4. 稳定性对比（成功率方差）
        ax4 = plt.subplot(2, 3, 4)
        stability_vars = [self.results[alg]['success_rate_var'] for alg in algs]
        
        bars = ax4.bar(algs, stability_vars, color='green')
        ax4.set_title('🎯 Stability Comparison (Lower Variance Better)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Success Rate Variance')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(stability_vars)*0.01,
                    f'{stability_vars[i]:.4f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 5. 综合评分雷达图
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        # 准备雷达图数据
        metrics = ['Success Rate', 'Efficiency', 'Stability']
        
        # 为每个算法计算归一化分数
        max_success = max([np.mean(self.results[alg]['successes']) for alg in algs])
        min_steps = min([np.mean(self.results[alg]['steps']) for alg in algs])
        max_steps = max([np.mean(self.results[alg]['steps']) for alg in algs])
        min_var = min([self.results[alg]['success_rate_var'] for alg in algs])
        max_var = max([self.results[alg]['success_rate_var'] for alg in algs])
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, alg in enumerate(algs[:6]):  # 最多显示6个算法
            # 计算归一化分数
            success_norm = np.mean(self.results[alg]['successes']) / max_success if max_success > 0 else 0
            efficiency_norm = (max_steps - np.mean(self.results[alg]['steps'])) / (max_steps - min_steps) if max_steps > min_steps else 0
            stability_norm = (max_var - self.results[alg]['success_rate_var']) / (max_var - min_var) if max_var > min_var else 0
            
            values = [success_norm, efficiency_norm, stability_norm]
            values += values[:1]  # 闭合
            
            ax5.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax5.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics)
        ax5.set_title('🕸️ Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. 学习曲线对比（如果有训练数据）
        ax6 = plt.subplot(2, 3, 6)
        
        # 绘制成功率随时间的变化（模拟学习曲线）
        for i, alg in enumerate(algs):
            # 使用成功率的累积平均作为学习曲线
            successes = self.results[alg]['successes']
            cumulative_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
            ax6.plot(cumulative_success, label=alg, color=colors[i % len(colors)])
        
        ax6.set_title('📈 Convergence Curve (Cumulative Success Rate)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Test Episode')
        ax6.set_ylabel('Cumulative Success Rate')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"algorithm_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📄 可视化图表已保存: {filename}")
        
        plt.show()
    
    def save_detailed_results(self):
        """保存详细结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备保存的数据
        save_data = {
            'timestamp': timestamp,
            'test_config': {
                'test_episodes': self.test_episodes,
                'stability_tests': self.stability_tests,
                'track_size': self.env.track_size,
                'max_speed': self.env.max_speed
            },
            'results': {}
        }
        
        # 处理每个算法的结果
        for alg_name, results in self.results.items():
            save_data['results'][alg_name] = {
                'success_rate': float(np.mean(results['successes'])),
                'avg_reward': float(np.mean(results['rewards'])),
                'avg_steps': float(np.mean(results['steps'])),
                'reward_std': float(np.std(results['rewards'])),
                'steps_std': float(np.std(results['steps'])),
                'success_rate_variance': float(results['success_rate_var']),
                'total_tests': len(results['successes']),
                'successful_tests': sum(results['successes'])
            }
        
        # 保存到JSON文件
        filename = f"comprehensive_comparison_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 详细结果已保存: {filename}")
    
    def run_full_comparison(self):
        """运行完整对比流程"""
        print("🚀 开始强化学习算法全面对比")
        print("="*60)
        
        start_time = time.time()
        
        # 1. 加载或训练所有算法
        self.load_or_train_all_algorithms()
        
        # 2. 全面性能测试
        self.comprehensive_performance_test()
        
        # 3. 生成对比报告
        df, comparison_data = self.generate_comprehensive_report()
        
        # 4. 创建可视化图表
        self.create_comprehensive_visualizations()
        
        # 5. 保存详细结果
        self.save_detailed_results()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 全面对比完成！")
        print(f"⏱️ 总耗时: {total_time:.2f}秒")
        print(f"📊 共测试了{len(self.algorithms)}个算法")
        print(f"🧪 每个算法进行了{self.test_episodes}次基础测试和{self.stability_tests}次稳定性测试")
        
        return self.results


def main():
    """主函数"""
    print("🔥 强化学习算法终极对比大战！")
    print("本次对比将全面评估以下算法:")
    print("  1. REINFORCE (策略梯度)")
    print("  2. Actor-Critic (Actor-Critic)")
    print("  3. PPO (近端策略优化)")
    print("  4. TRPO (信任区域策略优化)")
    print("  5. Q-Learning (值函数方法)")
    print("  6. Sarsa(λ) (值函数方法)")
    print("  7. Q-Guided AC (混合方法)")
    print("  8. Ultra Q-Guided AC (超级优化混合方法)")
    
    # 创建对比器
    comparator = AlgorithmComparator()
    
    # 运行完整对比
    results = comparator.run_full_comparison()
    
    print("\n" + "="*60)
    print("🏆 终极对比总结")
    print("="*60)
    
    # 找出最佳算法
    best_success = max(results.items(), key=lambda x: np.mean(x[1]['successes']))
    best_efficiency = min(results.items(), key=lambda x: np.mean(x[1]['steps']))
    best_stability = min(results.items(), key=lambda x: x[1]['success_rate_var'])
    
    print(f"🥇 最高成功率: {best_success[0]} ({np.mean(best_success[1]['successes']):.2%})")
    print(f"⚡ 最高效率: {best_efficiency[0]} ({np.mean(best_efficiency[1]['steps']):.1f}步)")
    print(f"🎯 最佳稳定性: {best_stability[0]} (方差: {best_stability[1]['success_rate_var']:.6f})")
    
    # 综合评分冠军
    composite_scores = comparator._calculate_composite_scores()
    champion = composite_scores[0]
    print(f"👑 综合冠军: {champion[0]} (得分: {champion[1]:.3f})")
    
    print(f"\n🎊 恭喜！对比分析完成。查看生成的图表和报告文件以获取详细信息。")


if __name__ == "__main__":
    main() 