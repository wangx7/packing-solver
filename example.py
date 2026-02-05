"""
矩形装箱求解器 - 使用示例
"""

from packing_solver import solve, can_fit, get_placement, visualize, SolveMode
import multiprocessing as mp

def main():
    print("=" * 60)
    print("矩形装箱求解器 - 使用示例")
    print("=" * 60)
    
    # ==========================================
    # 示例1: 最简单的用法 - 只判断能不能放
    # ==========================================
    print("\n【示例1】简单判断能否放入")
    print("-" * 40)
    
    # 定义容器尺寸
    容器宽 = 20
    容器高 = 15
    
    # 定义要放入的矩形 (宽, 高)
    矩形列表 = [(8, 6), (7, 5), (6, 4)]
    
    # 判断能否放入
    能放入 = can_fit(容器宽, 容器高, 矩形列表)
    
    print(f"容器: {容器宽} × {容器高}")
    print(f"矩形: {矩形列表}")
    print(f"结果: {'✓ 可以放入!' if 能放入 else '✗ 放不下'}")
    
    # ==========================================
    # 示例2: 获取具体放置方案
    # ==========================================
    print("\n【示例2】获取放置方案")
    print("-" * 40)
    
    W, H = 25, 20
    rects = [(10, 8), (9, 6), (7, 5)]
    
    # 获取放置方案
    placement = get_placement(W, H, rects, mode="precise")
    
    if placement:
        print(f"容器: {W} × {H}")
        print(f"矩形: {rects}")
        print("放置方案:")
        for i, p in enumerate(placement):
            print(f"  矩形{i+1} ({rects[i][0]}×{rects[i][1]}): "
                  f"中心点({p['cx']:.1f}, {p['cy']:.1f}), "
                  f"旋转{p['angle']*180/3.14159:.0f}°")
    else:
        print("无法放入")
    
    # ==========================================
    # 示例3: 使用不同模式
    # ==========================================
    print("\n【示例3】三种求解模式对比")
    print("-" * 40)
    
    W, H = 30, 25
    rects = [(12, 9), (10, 8), (9, 6), (8, 5), (7, 4)]
    
    print(f"容器: {W} × {H}")
    print(f"矩形数量: {len(rects)}")
    print()
    
    # 快速模式 - 最快，但可能漏解
    result = solve(W, H, rects, SolveMode.FAST)
    print(f"FAST模式:    {'✓' if result.success else '✗'} | "
          f"耗时: {result.time_ms:6.1f}ms | {result.confidence}")
    
    # 精确模式 - 较慢，但更准确
    result = solve(W, H, rects, SolveMode.PRECISE)
    print(f"PRECISE模式: {'✓' if result.success else '✗'} | "
          f"耗时: {result.time_ms:6.1f}ms | {result.confidence}")
    
    # ==========================================
    # 示例4: 控制并行
    # ==========================================
    print("\n【示例4】串行 vs 并行")
    print("-" * 40)
    
    W, H = 35, 28
    rects = [(12, 9), (10, 8), (9, 7), (8, 6), (7, 5), (6, 4)]
    
    # 串行
    result1 = solve(W, H, rects, SolveMode.PRECISE, parallel=False)
    print(f"串行: {'✓' if result1.success else '✗'} | 耗时: {result1.time_ms:.1f}ms")
    
    # 并行
    result2 = solve(W, H, rects, SolveMode.PRECISE, parallel=True)
    print(f"并行: {'✓' if result2.success else '✗'} | 耗时: {result2.time_ms:.1f}ms | "
          f"使用{result2.workers_used}个进程")
    
    # ==========================================
    # 示例5: 不可能的情况
    # ==========================================
    print("\n【示例5】检测不可能的情况")
    print("-" * 40)
    
    W, H = 10, 10
    rects = [(7, 7), (7, 7)]  # 两个7×7的矩形不可能放入10×10
    
    result = solve(W, H, rects, SolveMode.FAST)
    print(f"容器: {W} × {H}")
    print(f"矩形: {rects}")
    print(f"结果: {'✓ 可以' if result.success else '✗ 不可以'}")
    print(f"说明: {result.mode_used}")
    
    # ==========================================
    # 示例6: 可视化结果
    # ==========================================
    print("\n【示例6】可视化")
    print("-" * 40)
    
    W, H = 25, 20
    rects = [(10, 8), (9, 6), (7, 5), (6, 4)]
    
    result = solve(W, H, rects, SolveMode.PRECISE)
    
    if result.success:
        print("求解成功，正在显示可视化...")
        print("(如果没有显示图形窗口，请确保已安装 matplotlib)")
        try:
            visualize(W, H, result.placements, rects, 
                     title=f"装箱结果 (耗时 {result.time_ms:.0f}ms)")
        except Exception as e:
            print(f"可视化失败: {e}")
            print("请运行: pip install matplotlib")


if __name__ == "__main__":
    mp.freeze_support()
    main()
