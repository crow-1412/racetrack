import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from sarsa_lambda import SarsaLambdaAgent
from racetrack_env import RacetrackEnv

def train_and_find_successful_path():
    """训练智能体并找到成功路径"""
    print("=== 开始训练智能体并寻找成功路径 ===")
    
    # 创建环境
    env = RacetrackEnv(track_size=(32, 17), max_speed=5)
    print(f"环境信息：")
    print(f"  - 赛道大小: {env.track_size}")
    print(f"  - 起点数量: {len(env.start_positions)}")
    print(f"  - 终点数量: {len(env.goal_positions)}")
    print(f"  - 起点位置: {env.start_positions}")
    print(f"  - 终点位置: {env.goal_positions}")
    
    # 显示赛道结构
    print(f"\n=== 赛道结构分析 ===")
    track_info = {}
    for i in range(4):
        count = np.sum(env.track == i)
        track_info[i] = count
        type_name = ['空地', '墙壁', '起点', '终点'][i]
        print(f"  - {type_name}: {count} 个格子")
    
    # 创建智能体
    agent = SarsaLambdaAgent(
        env=env,
        alpha=0.3,      # 较高学习率
        gamma=0.95,     
        lambda_=0.8,    
        epsilon=0.3     # 较高探索率
    )
    
    print(f"\n=== 开始训练智能体 ===")
    # 训练智能体
    n_episodes = 3000  # 增加训练轮数
    rewards, steps = agent.train(n_episodes=n_episodes, verbose=True)
    
    print(f"\n=== 寻找成功路径 ===")
    # 寻找成功路径
    successful_paths = []
    max_attempts = 50
    
    # 设置测试时的探索率为0（纯贪婪策略）
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for attempt in range(max_attempts):
        reward, step_count, path = agent.test_episode(render=False)
        is_success = len(path) > 1 and path[-1] in env.goal_positions
        
        print(f"尝试 {attempt+1}: 步数={step_count}, 奖励={reward:.2f}, "
              f"成功={'是' if is_success else '否'}, 终点={path[-1] if path else 'N/A'}")
        
        if is_success:
            successful_paths.append({
                'reward': reward,
                'steps': step_count,
                'path': path,
                'start': path[0],
                'end': path[-1]
            })
            print(f"★ 找到成功路径 {len(successful_paths)}!")
            
            if len(successful_paths) >= 5:  # 找到5条成功路径
                break
    
    # 恢复原始探索率
    agent.epsilon = original_epsilon
    
    if successful_paths:
        print(f"\n=== 成功路径统计 ===")
        print(f"找到 {len(successful_paths)} 条成功路径")
        
        # 选择最短路径
        best_path_info = min(successful_paths, key=lambda x: x['steps'])
        print(f"最短路径: {best_path_info['steps']} 步, 奖励: {best_path_info['reward']:.2f}")
        
        # 选择奖励最高的路径
        best_reward_info = max(successful_paths, key=lambda x: x['reward'])
        print(f"最高奖励路径: {best_reward_info['steps']} 步, 奖励: {best_reward_info['reward']:.2f}")
        
        # 使用步数最少的路径
        selected_path = best_path_info
        
        print(f"\n=== 选定路径详情 ===")
        print(f"起点: {selected_path['start']}")
        print(f"终点: {selected_path['end']}")
        print(f"步数: {selected_path['steps']}")
        print(f"奖励: {selected_path['reward']:.2f}")
        
        # 保存路径到文件
        save_path_to_file(selected_path, successful_paths)
        
        # 绘制路径
        visualize_successful_path(env, selected_path)
        
        return selected_path, successful_paths
    else:
        print("❌ 未找到成功路径，智能体需要更多训练")
        return None, []

def save_path_to_file(selected_path, all_successful_paths):
    """保存路径到文件"""
    print(f"\n=== 保存路径到文件 ===")
    
    # 保存为JSON格式（人类可读）
    path_data = {
        'selected_path': selected_path,
        'all_successful_paths': all_successful_paths,
        'metadata': {
            'total_successful_paths': len(all_successful_paths),
            'selected_reason': 'shortest_steps'
        }
    }
    
    with open('successful_paths.json', 'w', encoding='utf-8') as f:
        json.dump(path_data, f, indent=2, ensure_ascii=False)
    
    # 保存为pickle格式（完整数据）
    with open('successful_paths.pkl', 'wb') as f:
        pickle.dump(path_data, f)
    
    print(f"路径数据已保存到:")
    print(f"  - successful_paths.json (人类可读)")
    print(f"  - successful_paths.pkl (完整数据)")
    
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

def visualize_successful_path(env, path_info):
    """可视化成功路径"""
    print(f"\n=== 开始绘制路径图 ===")
    
    path = path_info['path']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
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
        
        # 绘制路径线（更粗的线）
        ax.plot(path_y, path_x, 'b-', linewidth=5, alpha=0.8, label='Agent Path')
        
        # 标记起点（大绿点）
        ax.plot(path_y[0], path_x[0], 'go', markersize=15, label='Start', 
                markeredgecolor='darkgreen', markeredgewidth=3)
        
        # 标记终点（大红点）
        ax.plot(path_y[-1], path_x[-1], 'ro', markersize=15, label='Goal', 
                markeredgecolor='darkred', markeredgewidth=3)
        
        # 添加路径点编号（每10个点标一个）
        step_interval = max(1, len(path) // 10)
        for i in range(0, len(path), step_interval):
            ax.text(path_y[i], path_x[i], str(i), fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   ha='center', va='center', weight='bold')
        
        # 添加方向箭头（更多箭头）
        arrow_interval = max(1, len(path) // 15)
        for i in range(0, len(path)-1, arrow_interval):
            if i + 1 < len(path):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if dx != 0 or dy != 0:
                    ax.arrow(path_y[i], path_x[i], dy*0.6, dx*0.6, 
                            head_width=0.4, head_length=0.4, fc='blue', ec='darkblue', 
                            alpha=0.8, linewidth=2)
    
    # 设置标题和标签
    title = f'Racetrack Problem - Successful Path Visualization\n'
    title += f'Steps: {path_info["steps"]}, Reward: {path_info["reward"]:.2f}'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('Y Coordinate (Column)', fontsize=14)
    ax.set_ylabel('X Coordinate (Row)', fontsize=14)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, env.track.shape[1] - 0.5)
    ax.set_ylim(env.track.shape[0] - 0.5, -0.5)  # 反转y轴
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)
    
    # 添加详细信息文本框
    info_text = f"""Path Information:
• Start: {path[0]}
• Goal: {path[-1]}
• Total Steps: {len(path)}
• Reward: {path_info['reward']:.2f}

Legend:
• Light Green: Start Area
• Light Red: Goal Area  
• Light Gray: Track
• Black: Walls
• Blue Line: Agent Path"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig('successful_path_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('successful_path_detailed.pdf', bbox_inches='tight')  # 也保存PDF版本
    print("详细路径图已保存为:")
    print("  - successful_path_detailed.png (高清PNG)")
    print("  - successful_path_detailed.pdf (矢量PDF)")
    
    plt.show()

def load_and_visualize_saved_path():
    """加载保存的路径并可视化"""
    try:
        with open('successful_paths.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=== 加载保存的成功路径 ===")
        selected_path = data['selected_path']
        all_paths = data['all_successful_paths']
        
        print(f"已保存 {len(all_paths)} 条成功路径")
        print(f"选定路径: {selected_path['steps']} 步, 奖励: {selected_path['reward']:.2f}")
        
        # 重新创建环境进行可视化
        env = RacetrackEnv(track_size=(32, 17), max_speed=5)
        visualize_successful_path(env, selected_path)
        
        return selected_path
        
    except FileNotFoundError:
        print("未找到保存的路径文件，请先运行训练程序")
        return None

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 训练智能体并寻找成功路径")
    print("2. 加载已保存的路径并可视化")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        successful_path, all_paths = train_and_find_successful_path()
    elif choice == "2":
        successful_path = load_and_visualize_saved_path()
    else:
        print("无效选择，默认运行训练模式")
        successful_path, all_paths = train_and_find_successful_path() 