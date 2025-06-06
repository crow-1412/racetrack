import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import torch
from actor_critic_improved_fixed import OptimizedActorCriticAgent
from racetrack_env import RacetrackEnv

def load_and_test_actor_critic():
    """加载训练好的Actor-Critic模型并测试"""
    print("=== 加载Actor-Critic模型并寻找成功路径 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    print(f"  - 起点位置: {env.start_positions}")
    print(f"  - 终点位置: {env.goal_positions}")
    
    # 创建智能体
    agent = OptimizedActorCriticAgent(
        env=env,
        lr=0.001,
        gamma=0.99,
        hidden_dim=128,
        buffer_size=128,
        gae_lambda=0.95
    )
    
    # 加载训练好的模型
    try:
        print("\n=== 加载训练好的模型 ===")
        agent.load_model('fixed_actor_critic_model.pth')
        print("✅ 成功加载模型: fixed_actor_critic_model.pth")
    except FileNotFoundError:
        print("❌ 未找到模型文件，请先训练模型")
        return None, []
    
    print(f"\n=== 寻找Actor-Critic成功路径 ===")
    # 寻找成功路径
    successful_paths = []
    max_attempts = 100  # 增加尝试次数
    
    for attempt in range(max_attempts):
        reward, step_count, path, success = agent.test_episode(render=False)
        
        print(f"尝试 {attempt+1:3d}: 步数={step_count:3d}, 奖励={reward:7.2f}, "
              f"成功={'是' if success else '否'}, 终点={path[-1] if path else 'N/A'}")
        
        if success:
            successful_paths.append({
                'reward': reward,
                'steps': step_count,
                'path': path,
                'start': path[0],
                'end': path[-1]
            })
            print(f"🎯 找到Actor-Critic成功路径 {len(successful_paths)}!")
            
            if len(successful_paths) >= 10:  # 找到10条成功路径
                break
    
    if successful_paths:
        print(f"\n=== Actor-Critic成功路径统计 ===")
        print(f"找到 {len(successful_paths)} 条成功路径")
        
        # 选择最短路径
        best_path_info = min(successful_paths, key=lambda x: x['steps'])
        print(f"最短路径: {best_path_info['steps']} 步, 奖励: {best_path_info['reward']:.2f}")
        
        # 选择奖励最高的路径
        best_reward_info = max(successful_paths, key=lambda x: x['reward'])
        print(f"最高奖励路径: {best_reward_info['steps']} 步, 奖励: {best_reward_info['reward']:.2f}")
        
        # 使用步数最少的路径
        selected_path = best_path_info
        
        print(f"\n=== 选定Actor-Critic路径详情 ===")
        print(f"起点: {selected_path['start']}")
        print(f"终点: {selected_path['end']}")
        print(f"步数: {selected_path['steps']}")
        print(f"奖励: {selected_path['reward']:.2f}")
        
        # 保存路径到文件
        save_actor_critic_path_to_file(selected_path, successful_paths)
        
        # 绘制路径
        visualize_actor_critic_path(env, selected_path)
        
        return selected_path, successful_paths
    else:
        print("❌ 未找到成功路径，Actor-Critic模型可能需要更多训练")
        
        # 即使没有成功路径，也展示一个最好的尝试
        print("\n=== 展示最佳尝试路径 ===")
        best_attempt = None
        best_reward = float('-inf')
        
        for attempt in range(20):  # 再尝试20次找最佳路径
            reward, step_count, path, success = agent.test_episode(render=False)
            if reward > best_reward:
                best_reward = reward
                best_attempt = {
                    'reward': reward,
                    'steps': step_count,
                    'path': path,
                    'start': path[0] if path else None,
                    'end': path[-1] if path else None,
                    'success': success
                }
        
        if best_attempt:
            print(f"最佳尝试: {best_attempt['steps']} 步, 奖励: {best_attempt['reward']:.2f}")
            visualize_actor_critic_path(env, best_attempt, is_success=False)
        
        return None, []

def save_actor_critic_path_to_file(selected_path, all_successful_paths):
    """保存Actor-Critic路径到文件"""
    print(f"\n=== 保存Actor-Critic路径到文件 ===")
    
    # 保存为JSON格式（人类可读）
    path_data = {
        'algorithm': 'Actor-Critic (Fixed)',
        'selected_path': selected_path,
        'all_successful_paths': all_successful_paths,
        'metadata': {
            'total_successful_paths': len(all_successful_paths),
            'selected_reason': 'shortest_steps',
            'model_file': 'fixed_actor_critic_model.pth'
        }
    }
    
    with open('actor_critic_successful_paths.json', 'w', encoding='utf-8') as f:
        json.dump(path_data, f, indent=2, ensure_ascii=False)
    
    # 保存为pickle格式（完整数据）
    with open('actor_critic_successful_paths.pkl', 'wb') as f:
        pickle.dump(path_data, f)
    
    print(f"Actor-Critic路径数据已保存到:")
    print(f"  - actor_critic_successful_paths.json (人类可读)")
    print(f"  - actor_critic_successful_paths.pkl (完整数据)")
    
    # 打印路径详情
    print(f"\n=== 完整路径坐标 ===")
    path = selected_path['path']
    print(f"路径长度: {len(path)} 个点")
    print("路径坐标:")
    for i, (x, y) in enumerate(path):
        if i % 5 == 0:  # 每行显示5个点
            print()
            print(f"  {i:2d}-{min(i+4, len(path)-1):2d}: ", end="")
        print(f"({x:2d},{y:2d})", end=" ")
    print()

def visualize_actor_critic_path(env, path_info, is_success=True):
    """可视化Actor-Critic路径"""
    print(f"\n=== 开始绘制Actor-Critic路径图 ===")
    
    path = path_info['path']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(18, 14))  # 更大的图片
    
    # 创建颜色映射
    track_colors = np.zeros((*env.track.shape, 3))
    
    for i in range(env.track.shape[0]):
        for j in range(env.track.shape[1]):
            if env.track[i, j] == 0:  # 空地
                track_colors[i, j] = [0.95, 0.95, 0.95]  # 浅灰色
            elif env.track[i, j] == 1:  # 墙
                track_colors[i, j] = [0.1, 0.1, 0.1]  # 深黑色
            elif env.track[i, j] == 2:  # 起点
                track_colors[i, j] = [0.5, 0.9, 0.5]  # 浅绿色
            elif env.track[i, j] == 3:  # 终点
                track_colors[i, j] = [0.9, 0.5, 0.5]  # 浅红色
    
    # 显示赛道
    ax.imshow(track_colors, origin='upper', aspect='equal')
    
    # 绘制网格
    for i in range(env.track.shape[0] + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
    for j in range(env.track.shape[1] + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.5)
    
    # 绘制路径
    if path and len(path) > 1:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        # 根据是否成功选择颜色
        if is_success:
            path_color = 'orange'
            path_label = 'Actor-Critic Successful Path'
            edge_color = 'darkorange'
        else:
            path_color = 'red'
            path_label = 'Actor-Critic Best Attempt'
            edge_color = 'darkred'
        
        # 绘制路径线（粗线，使用橙色表示Actor-Critic）
        ax.plot(path_y, path_x, path_color, linewidth=6, alpha=0.9, label=path_label)
        
        # 标记起点（大绿点）
        ax.plot(path_y[0], path_x[0], 'go', markersize=18, label='Start', 
                markeredgecolor='darkgreen', markeredgewidth=4)
        
        # 标记终点（大红点或目标点）
        if is_success:
            ax.plot(path_y[-1], path_x[-1], 'ro', markersize=18, label='Goal Reached', 
                    markeredgecolor='darkred', markeredgewidth=4)
        else:
            ax.plot(path_y[-1], path_x[-1], 'yo', markersize=18, label='Final Position', 
                    markeredgecolor='orange', markeredgewidth=4)
        
        # 添加路径点编号（每8个点标一个）
        step_interval = max(1, len(path) // 12)
        for i in range(0, len(path), step_interval):
            ax.text(path_y[i], path_x[i], str(i), fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
                   ha='center', va='center', weight='bold')
        
        # 添加方向箭头（更多箭头）
        arrow_interval = max(1, len(path) // 20)
        for i in range(0, len(path)-1, arrow_interval):
            if i + 1 < len(path):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if dx != 0 or dy != 0:
                    ax.arrow(path_y[i], path_x[i], dy*0.7, dx*0.7, 
                            head_width=0.5, head_length=0.5, fc=path_color, ec=edge_color, 
                            alpha=0.8, linewidth=2.5)
    
    # 设置标题和标签
    status = "Successful" if is_success else "Best Attempt"
    title = f'Racetrack Problem - Actor-Critic {status} Path Visualization\n'
    title += f'Steps: {path_info["steps"]}, Reward: {path_info["reward"]:.2f}'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=25)
    
    ax.set_xlabel('Y Coordinate (Column)', fontsize=16)
    ax.set_ylabel('X Coordinate (Row)', fontsize=16)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, env.track.shape[1] - 0.5)
    ax.set_ylim(env.track.shape[0] - 0.5, -0.5)  # 反转y轴
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=14)
    
    # 添加详细信息文本框
    success_text = "SUCCESS!" if is_success else "BEST ATTEMPT"
    info_text = f"""Actor-Critic Path Information:
• Algorithm: Actor-Critic (Fixed)
• Status: {success_text}
• Start: {path[0]}
• End: {path[-1]}
• Total Steps: {len(path)}
• Reward: {path_info['reward']:.2f}

Legend:
• Light Green: Start Area
• Light Red: Goal Area  
• Light Gray: Track
• Black: Walls
• Orange Line: Actor-Critic Path"""
    
    box_color = "lightgreen" if is_success else "lightcoral"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, alpha=0.9))
    
    plt.tight_layout()
    
    # 保存高质量图片
    filename = 'actor_critic_successful_path.png' if is_success else 'actor_critic_best_attempt.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Actor-Critic路径图已保存为: {filename}")
    
    plt.show()

def load_and_visualize_saved_actor_critic_path():
    """加载保存的Actor-Critic路径并可视化"""
    try:
        with open('actor_critic_successful_paths.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=== 加载保存的Actor-Critic成功路径 ===")
        selected_path = data['selected_path']
        all_paths = data['all_successful_paths']
        
        print(f"算法: {data.get('algorithm', 'Actor-Critic')}")
        print(f"已保存 {len(all_paths)} 条成功路径")
        print(f"选定路径: {selected_path['steps']} 步, 奖励: {selected_path['reward']:.2f}")
        
        # 重新创建环境进行可视化
        env = RacetrackEnv(track_size=(32, 17), max_speed=5)
        visualize_actor_critic_path(env, selected_path)
        
        return selected_path
        
    except FileNotFoundError:
        print("未找到保存的Actor-Critic路径文件，请先运行测试程序")
        return None

def compare_all_algorithms():
    """比较所有算法的结果"""
    print("=== 比较所有算法的结果 ===")
    
    algorithms = {
        'Q-learning': 'qlearning_successful_paths.json',
        'Sarsa(λ)': 'successful_paths.json', 
        'Actor-Critic': 'actor_critic_successful_paths.json'
    }
    
    results = {}
    
    for alg_name, filename in algorithms.items():
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results[alg_name] = data
            print(f"✓ 成功加载{alg_name}结果")
        except FileNotFoundError:
            print(f"✗ 未找到{alg_name}结果文件: {filename}")
    
    if len(results) >= 2:
        print(f"\n=== 算法性能比较 ===")
        
        for alg_name, data in results.items():
            path = data['selected_path']
            print(f"{alg_name}:")
            print(f"  - 步数: {path['steps']}")
            print(f"  - 奖励: {path['reward']:.2f}")
            print(f"  - 成功路径数: {len(data['all_successful_paths'])}")
        
        # 找出最佳性能
        best_steps = min(results.values(), key=lambda x: x['selected_path']['steps'])
        best_reward = max(results.values(), key=lambda x: x['selected_path']['reward'])
        most_paths = max(results.values(), key=lambda x: len(x['all_successful_paths']))
        
        print(f"\n=== 最佳性能 ===")
        for alg_name, data in results.items():
            if data == best_steps:
                print(f"🏆 最短路径: {alg_name} ({data['selected_path']['steps']} 步)")
            if data == best_reward:
                print(f"🏆 最高奖励: {alg_name} ({data['selected_path']['reward']:.2f})")
            if data == most_paths:
                print(f"🏆 最多成功路径: {alg_name} ({len(data['all_successful_paths'])} 条)")
    
    elif len(results) == 1:
        alg_name = list(results.keys())[0]
        print(f"只有{alg_name}结果可用")
    else:
        print("没有可比较的结果")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 加载Actor-Critic模型并寻找成功路径")
    print("2. 加载已保存的Actor-Critic路径并可视化")
    print("3. 比较所有算法的结果")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        successful_path, all_paths = load_and_test_actor_critic()
    elif choice == "2":
        successful_path = load_and_visualize_saved_actor_critic_path()
    elif choice == "3":
        compare_all_algorithms()
    else:
        print("无效选择，默认运行模型测试模式")
        successful_path, all_paths = load_and_test_actor_critic() 