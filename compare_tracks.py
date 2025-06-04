#!/usr/bin/env python3

from racetrack_env import RacetrackEnv
from simple_racetrack_env import SimpleRacetrackEnv

def print_track_matrix(track, title, size):
    """打印赛道矩阵"""
    print(f"\n{title}")
    print("=" * 60)
    print("  0=道路, 1=墙壁, 2=起点, 3=终点")
    print("=" * 60)
    
    # 添加列标题
    col_header = "     " + " ".join(f"{i:2d}" for i in range(size[1]))
    print(col_header)
    print("     " + "---" * size[1])
    
    for i, row in enumerate(track):
        row_str = " ".join(str(cell) for cell in row)
        print(f"行{i:2d}:  {row_str}")
    
    print("=" * 60)

def compare_tracks():
    """比较原始赛道和简化赛道"""
    print("🏁 原始赛道 vs 简化赛道对比分析")
    
    # 创建环境
    original_env = RacetrackEnv()
    simple_env = SimpleRacetrackEnv()
    
    # 显示基本信息对比
    print("\n📊 基本参数对比:")
    print(f"{'参数':<15} {'原始赛道':<15} {'简化赛道':<15}")
    print("-" * 50)
    print(f"{'赛道大小':<15} {str(original_env.track_size):<15} {str(simple_env.track_size):<15}")
    print(f"{'最大速度':<15} {original_env.max_speed:<15} {simple_env.max_speed:<15}")
    print(f"{'状态空间':<15} {original_env.get_state_space_size():<15,} {simple_env.get_state_space_size():<15,}")
    print(f"{'起点数量':<15} {len(original_env.start_positions):<15} {len(simple_env.start_positions):<15}")
    print(f"{'终点数量':<15} {len(original_env.goal_positions):<15} {len(simple_env.goal_positions):<15}")
    
    # 显示矩阵布局
    print_track_matrix(original_env.track, "🔹 原始赛道布局 (32x17)", original_env.track_size)
    print_track_matrix(simple_env.track, "🔸 简化赛道布局 (20x10)", simple_env.track_size)
    
    # 分析差异
    print("\n🔍 布局差异分析:")
    print("原始赛道特点:")
    print("  - L型赛道，从底部(第31行)起点向上，然后向右到达顶部右侧终点")
    print("  - 垂直段：逐渐扩宽，从第0列的部分区域到第3-8列的全通道")
    print("  - 水平段：第0-5行的第10-16列")
    print("  - 起点：整个第31行(17个起点)")
    print("  - 终点：第16列的前6行")
    
    print("\n简化赛道特点:")
    print("  - 保持L型结构，从底部(第19行)起点向上，然后向右到达顶部右侧终点")
    print("  - 垂直段：第10-19行的第0-5列")
    print("  - 转弯段：第6-10行的第2-7列")
    print("  - 水平段：第0-6行的第4-9列")
    print("  - 起点：第19行的前5列")
    print("  - 终点：第9列的前5行")
    
    # 相似性分析
    print("\n✅ 相似性:")
    print("  1. 都是L型赛道布局")
    print("  2. 都是从底部起点向上再向右到达终点")
    print("  3. 都有逐渐扩宽的通道设计")
    print("  4. 起点和终点都在赛道的对角位置")
    
    # 差异分析
    print("\n❗ 主要差异:")
    print("  1. 尺寸缩小: 32x17 → 20x10 (面积减少约65%)")
    print("  2. 速度降低: 最大速度 5 → 2")
    print("  3. 状态空间: 65,824 → 5,000 (减少92%)")
    print("  4. 起点终点: 17个起点+6个终点 → 5个起点+5个终点")
    print("  5. 赛道细节: 简化版的转弯段和通道设计更紧凑")
    
    reduction_ratio = original_env.get_state_space_size() / simple_env.get_state_space_size()
    print(f"\n📈 总体简化程度: 状态空间减少 {reduction_ratio:.1f} 倍")

if __name__ == "__main__":
    compare_tracks() 