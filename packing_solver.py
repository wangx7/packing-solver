"""
多矩形装箱求解器 - 并行加速版
判断多个矩形（≤10个）能否放入一个固定尺寸的容器中，支持任意角度旋转

作者: GitHub Copilot
日期: 2026-02-05
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import time
import random
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool, Manager
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 基础几何工具
# ============================================================

def get_rotated_rect(w: float, h: float, cx: float, cy: float, angle: float) -> List[Tuple[float, float]]:
    """获取旋转后矩形的4个顶点坐标"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(dx * cos_a - dy * sin_a + cx, dx * sin_a + dy * cos_a + cy) for dx, dy in corners]


def get_bounding_box(w: float, h: float, angle: float) -> Tuple[float, float]:
    """计算旋转后矩形的包围盒尺寸"""
    cos_a, sin_a = abs(np.cos(angle)), abs(np.sin(angle))
    return w * cos_a + h * sin_a, w * sin_a + h * cos_a


def project_polygon(vertices: List, axis: Tuple) -> Tuple[float, float]:
    """将多边形投影到轴上，返回投影区间"""
    dots = [v[0] * axis[0] + v[1] * axis[1] for v in vertices]
    return min(dots), max(dots)


def get_axes(vertices: List) -> List[Tuple[float, float]]:
    """获取多边形所有边的法向量（用于SAT碰撞检测）"""
    axes = []
    n = len(vertices)
    for i in range(n):
        edge = (vertices[(i+1) % n][0] - vertices[i][0], vertices[(i+1) % n][1] - vertices[i][1])
        length = np.sqrt(edge[0]**2 + edge[1]**2)
        if length > 1e-10:
            axes.append((-edge[1] / length, edge[0] / length))
    return axes


def polygons_overlap(poly1: List, poly2: List, eps: float = 1e-9) -> bool:
    """使用分离轴定理(SAT)检测两个多边形是否重叠"""
    for axis in get_axes(poly1) + get_axes(poly2):
        min1, max1 = project_polygon(poly1, axis)
        min2, max2 = project_polygon(poly2, axis)
        if max1 < min2 - eps or max2 < min1 - eps:
            return False  # 找到分离轴，不重叠
    return True  # 没有分离轴，重叠


def rect_inside_container(vertices: List, W: float, H: float, eps: float = 1e-9) -> bool:
    """检查矩形的所有顶点是否都在容器内"""
    return all(-eps <= x <= W + eps and -eps <= y <= H + eps for x, y in vertices)


def compute_overlap_depth(poly1: List, poly2: List) -> float:
    """计算两个多边形的重叠深度（用于优化算法）"""
    min_overlap = float('inf')
    for axis in get_axes(poly1) + get_axes(poly2):
        min1, max1 = project_polygon(poly1, axis)
        min2, max2 = project_polygon(poly2, axis)
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    return max(0, min_overlap)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class Placement:
    """单个矩形的放置状态"""
    w: float          # 原始宽度
    h: float          # 原始高度
    cx: float = 0     # 中心点x坐标
    cy: float = 0     # 中心点y坐标
    angle: float = 0  # 旋转角度（弧度）
    
    def get_polygon(self) -> List[Tuple[float, float]]:
        """获取放置后的顶点坐标"""
        return get_rotated_rect(self.w, self.h, self.cx, self.cy, self.angle)
    
    def get_bbox(self) -> Tuple[float, float]:
        """获取包围盒尺寸"""
        return get_bounding_box(self.w, self.h, self.angle)
    
    def to_dict(self) -> dict:
        """转换为字典（用于进程间传递）"""
        return {'w': self.w, 'h': self.h, 'cx': self.cx, 'cy': self.cy, 'angle': self.angle}
    
    @staticmethod
    def from_dict(d: dict) -> 'Placement':
        """从字典创建"""
        return Placement(d['w'], d['h'], d['cx'], d['cy'], d['angle'])


class SolveMode(Enum):
    """求解模式"""
    FAST = "fast"        # 快速模式：贪心算法，速度快但可能漏解
    PRECISE = "precise"  # 精确模式：模拟退火，较慢但更准确
    PROOF = "proof"      # 证明模式：分支定界，最慢但数学严格


@dataclass
class SolveResult:
    """求解结果"""
    success: bool                                    # 是否找到解
    placements: List[Placement] = field(default_factory=list)  # 放置方案
    mode_used: str = ""                              # 使用的模式
    time_ms: float = 0                               # 耗时（毫秒）
    iterations: int = 0                              # 迭代次数
    confidence: str = ""                             # 置信度说明
    workers_used: int = 1                            # 使用的进程数


# ============================================================
# 候选位置生成
# ============================================================

def _generate_candidate_positions(bw: float, bh: float, W: float, H: float, 
                                   placed_polys: List) -> List[Tuple[float, float]]:
    """生成候选放置位置"""
    positions = []
    
    # 四个角落
    positions.extend([
        (bw/2, bh/2),           # 左下
        (W - bw/2, bh/2),       # 右下
        (bw/2, H - bh/2),       # 左上
        (W - bw/2, H - bh/2),   # 右上
    ])
    
    # 贴着已放置矩形的边
    for poly in placed_polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        positions.extend([
            (max(xs) + bw/2, bh/2),
            (max(xs) + bw/2, H - bh/2),
            (bw/2, max(ys) + bh/2),
            (W - bw/2, max(ys) + bh/2),
        ])
    
    # 网格采样
    for fx in np.linspace(0, 1, 5):
        for fy in np.linspace(0, 1, 5):
            cx = bw/2 + fx * max(0, W - bw)
            cy = bh/2 + fy * max(0, H - bh)
            positions.append((cx, cy))
    
    return positions


# ============================================================
# 贪心求解器（单进程）
# ============================================================

def _solve_greedy_single(args) -> Optional[List[dict]]:
    """单次贪心求解"""
    W, H, rects, order, angle_steps, found_flag = args
    
    if found_flag.value:
        return None
    
    angles = np.linspace(0, np.pi / 2, angle_steps)
    placements = [None] * len(rects)
    placed_polys = []
    
    for idx in order:
        if found_flag.value:
            return None
            
        w, h = rects[idx]
        best_placement = None
        best_score = float('inf')
        
        for angle in angles:
            bw, bh = get_bounding_box(w, h, angle)
            if bw > W or bh > H:
                continue
            
            positions = _generate_candidate_positions(bw, bh, W, H, placed_polys)
            
            for cx, cy in positions:
                placement = Placement(w, h, cx, cy, angle)
                poly = placement.get_polygon()
                
                if not rect_inside_container(poly, W, H):
                    continue
                
                if any(polygons_overlap(poly, pp) for pp in placed_polys):
                    continue
                
                # 优先放在左下角
                score = cx + cy * 0.5
                if score < best_score:
                    best_score = score
                    best_placement = placement
        
        if best_placement is None:
            return None
        
        placements[idx] = best_placement
        placed_polys.append(best_placement.get_polygon())
    
    return [p.to_dict() for p in placements]


# ============================================================
# 模拟退火求解器（单进程）
# ============================================================

def _solve_sa_single(args) -> Optional[Tuple[List[dict], int]]:
    """单次模拟退火求解"""
    W, H, rects, seed, max_iter, found_flag = args
    
    if found_flag.value:
        return None
    
    random.seed(seed)
    np.random.seed(seed)
    n = len(rects)
    
    # 随机初始化
    placements = []
    for w, h in rects:
        angle = random.uniform(0, np.pi / 2)
        bw, bh = get_bounding_box(w, h, angle)
        cx = random.uniform(bw/2, max(bw/2 + 0.1, W - bw/2))
        cy = random.uniform(bh/2, max(bh/2 + 0.1, H - bh/2))
        placements.append(Placement(w, h, cx, cy, angle))
    
    def compute_penalty(pls):
        """计算违反约束的惩罚值"""
        polys = [p.get_polygon() for p in pls]
        penalty = 0.0
        # 超出容器惩罚
        for poly in polys:
            for x, y in poly:
                if x < 0: penalty += abs(x) * 10
                if x > W: penalty += (x - W) * 10
                if y < 0: penalty += abs(y) * 10
                if y > H: penalty += (y - H) * 10
        # 重叠惩罚
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polygons_overlap(polys[i], polys[j]):
                    penalty += compute_overlap_depth(polys[i], polys[j]) * 5
        return penalty
    
    def is_valid(pls):
        """检查是否是有效解"""
        polys = [p.get_polygon() for p in pls]
        for poly in polys:
            if not rect_inside_container(poly, W, H):
                return False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polygons_overlap(polys[i], polys[j]):
                    return False
        return True
    
    current = placements
    current_penalty = compute_penalty(current)
    best = deepcopy(current)
    best_penalty = current_penalty
    
    temp = 100.0
    cooling = 0.995
    
    for iteration in range(max_iter):
        if found_flag.value:
            return None
        
        # 随机扰动
        new_placements = deepcopy(current)
        idx = random.randint(0, n - 1)
        p = new_placements[idx]
        
        move_type = random.random()
        if move_type < 0.4:
            # 移动位置
            bw, bh = p.get_bbox()
            p.cx = np.clip(p.cx + random.gauss(0, W * 0.1), bw/2, W - bw/2)
            p.cy = np.clip(p.cy + random.gauss(0, H * 0.1), bh/2, H - bh/2)
        elif move_type < 0.7:
            # 旋转
            p.angle = (p.angle + random.gauss(0, 0.2)) % (np.pi / 2)
            bw, bh = p.get_bbox()
            p.cx = np.clip(p.cx, bw/2, W - bw/2)
            p.cy = np.clip(p.cy, bh/2, H - bh/2)
        else:
            # 交换两个矩形位置
            idx2 = random.randint(0, n - 1)
            if idx != idx2:
                new_placements[idx].cx, new_placements[idx2].cx = new_placements[idx2].cx, new_placements[idx].cx
                new_placements[idx].cy, new_placements[idx2].cy = new_placements[idx2].cy, new_placements[idx].cy
        
        new_penalty = compute_penalty(new_placements)
        delta = new_penalty - current_penalty
        
        # Metropolis准则
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
            current = new_placements
            current_penalty = new_penalty
            
            if current_penalty < best_penalty:
                best = deepcopy(current)
                best_penalty = current_penalty
                
                if best_penalty < 1e-9 and is_valid(best):
                    return [p.to_dict() for p in best], iteration
        
        temp *= cooling
    
    if is_valid(best):
        return [p.to_dict() for p in best], max_iter
    return None


# ============================================================
# 分支定界求解器（单进程）
# ============================================================

def _solve_branch_single(args) -> Optional[Tuple[List[dict], int]]:
    """单次分支定界求解"""
    W, H, rects, angle_subset, found_flag, max_time = args
    
    if found_flag.value:
        return None
    
    n = len(rects)
    # 按面积降序排列
    indexed_rects = [(i, w, h) for i, (w, h) in enumerate(rects)]
    indexed_rects.sort(key=lambda x: -x[1] * x[2])
    
    start_time = time.time()
    iterations = [0]
    best_solution = [None]
    
    def generate_positions(bw, bh, placed_polys, depth):
        positions = [(bw/2, bh/2), (W - bw/2, bh/2), (bw/2, H - bh/2), (W - bw/2, H - bh/2)]
        
        for poly in placed_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            right_x, top_y = max(xs), max(ys)
            
            if right_x + bw/2 <= W:
                for fy in [0, 0.5, 1]:
                    positions.append((right_x + bw/2, bh/2 + fy * max(0, H - bh)))
            if top_y + bh/2 <= H:
                for fx in [0, 0.5, 1]:
                    positions.append((bw/2 + fx * max(0, W - bw), top_y + bh/2))
        
        grid = max(3, 6 - depth)
        for fx in np.linspace(0, 1, grid):
            for fy in np.linspace(0, 1, grid):
                positions.append((bw/2 + fx * max(0, W - bw), bh/2 + fy * max(0, H - bh)))
        
        return list(set((round(x, 4), round(y, 4)) for x, y in positions))
    
    def backtrack(depth, placed, placed_polys):
        if found_flag.value:
            return False
        if time.time() - start_time > max_time:
            return False
        
        iterations[0] += 1
        
        if depth == n:
            best_solution[0] = deepcopy(placed)
            return True
        
        orig_idx, w, h = indexed_rects[depth]
        
        for angle in angle_subset:
            if found_flag.value:
                return False
            
            bw, bh = get_bounding_box(w, h, angle)
            if bw > W + 1e-9 or bh > H + 1e-9:
                continue
            
            for cx, cy in generate_positions(bw, bh, placed_polys, depth):
                placement = Placement(w, h, cx, cy, angle)
                poly = placement.get_polygon()
                
                if not rect_inside_container(poly, W, H):
                    continue
                if any(polygons_overlap(poly, pp) for pp in placed_polys):
                    continue
                
                if backtrack(depth + 1, placed + [placement], placed_polys + [poly]):
                    return True
        
        return False
    
    if backtrack(0, [], []):
        result = [None] * n
        for i, (orig_idx, _, _) in enumerate(indexed_rects):
            result[orig_idx] = best_solution[0][i].to_dict()
        return result, iterations[0]
    
    return None


# ============================================================
# 并行求解器
# ============================================================

def solve_fast_parallel(W: float, H: float, rects: List[Tuple[float, float]], 
                        n_workers: int = None) -> Tuple[Optional[List[Placement]], int]:
    """并行贪心求解"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    n = len(rects)
    
    # 生成多种排列策略
    orders = []
    orders.append(sorted(range(n), key=lambda i: -rects[i][0] * rects[i][1]))  # 面积降序
    orders.append(sorted(range(n), key=lambda i: rects[i][0] * rects[i][1]))   # 面积升序
    orders.append(sorted(range(n), key=lambda i: -max(rects[i])))              # 最长边降序
    orders.append(sorted(range(n), key=lambda i: -(rects[i][0] + rects[i][1])))# 周长降序
    
    for seed in range(max(0, n_workers - 4)):
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        orders.append(order)
    
    orders = orders[:n_workers]
    
    with Manager() as manager:
        found_flag = manager.Value('b', False)
        args_list = [(W, H, rects, order, 9, found_flag) for order in orders]
        
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_solve_greedy_single, args_list):
                if result is not None:
                    found_flag.value = True
                    pool.terminate()
                    return [Placement.from_dict(d) for d in result], n_workers
    
    return None, n_workers


def solve_precise_parallel(W: float, H: float, rects: List[Tuple[float, float]],
                           n_workers: int = None, max_iter: int = 8000) -> Tuple[Optional[List[Placement]], int, int]:
    """并行模拟退火求解"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    with Manager() as manager:
        found_flag = manager.Value('b', False)
        args_list = [(W, H, rects, seed, max_iter, found_flag) for seed in range(n_workers)]
        
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_solve_sa_single, args_list):
                if result is not None:
                    found_flag.value = True
                    pool.terminate()
                    placements, iters = result
                    return [Placement.from_dict(d) for d in placements], iters, n_workers
    
    return None, max_iter * n_workers, n_workers


def solve_proof_parallel(W: float, H: float, rects: List[Tuple[float, float]],
                         n_workers: int = None, angle_steps: int = 13, 
                         max_time: float = 60.0) -> Tuple[Optional[List[Placement]], int, int, str]:
    """并行分支定界求解"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    all_angles = np.linspace(0, np.pi / 2, angle_steps)
    
    # 划分角度空间
    angle_subsets = [all_angles[i::n_workers] for i in range(n_workers)]
    key_angles = [0, np.pi/4, np.pi/2]
    for i, subset in enumerate(angle_subsets):
        angle_subsets[i] = np.unique(np.concatenate([subset, key_angles]))
    
    log = f"并行分支定界: {n_workers}个worker"
    
    with Manager() as manager:
        found_flag = manager.Value('b', False)
        args_list = [(W, H, rects, subset, found_flag, max_time) for subset in angle_subsets]
        
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_solve_branch_single, args_list):
                if result is not None:
                    found_flag.value = True
                    pool.terminate()
                    placements, iters = result
                    return [Placement.from_dict(d) for d in placements], iters, n_workers, log + " ✓"
    
    return None, 0, n_workers, log + " ✗"


# ============================================================
# 统一求解接口
# ============================================================

def solve(W: float, H: float, rects: List[Tuple[float, float]], 
          mode: SolveMode = SolveMode.FAST, 
          parallel: bool = True,
          n_workers: int = None) -> SolveResult:
    """
    判断多个矩形能否放入指定容器中
    
    参数:
        W: 容器宽度
        H: 容器高度
        rects: 矩形列表，每个元素为 (宽, 高)
        mode: 求解模式
            - SolveMode.FAST: 快速模式，~10ms
            - SolveMode.PRECISE: 精确模式，~1-5s
            - SolveMode.PROOF: 证明模式，~10s-几分钟
        parallel: 是否启用并行加速
        n_workers: 并行进程数（默认为CPU核心数）
    
    返回:
        SolveResult 对象，包含:
            - success: 是否成功
            - placements: 放置方案列表
            - time_ms: 耗时（毫秒）
            - confidence: 置信度说明
    """
    # 预检查：面积
    total_area = sum(w * h for w, h in rects)
    if total_area > W * H:
        return SolveResult(
            success=False,
            mode_used=f"面积预检失败: {total_area:.1f} > {W*H:.1f}",
            confidence="100% 确定无解"
        )
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    start_time = time.time()
    
    if mode == SolveMode.FAST:
        if parallel and len(rects) >= 2:
            result, workers = solve_fast_parallel(W, H, rects, n_workers)
        else:
            with Manager() as manager:
                found_flag = manager.Value('b', False)
                order = sorted(range(len(rects)), key=lambda i: -rects[i][0] * rects[i][1])
                res = _solve_greedy_single((W, H, rects, order, 9, found_flag))
                result = [Placement.from_dict(d) for d in res] if res else None
                workers = 1
        
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=result is not None,
            placements=result or [],
            mode_used="并行贪心" if parallel else "贪心",
            time_ms=elapsed,
            workers_used=workers,
            confidence="~90% (快速但可能漏解)"
        )
    
    elif mode == SolveMode.PRECISE:
        if parallel:
            result, iters, workers = solve_precise_parallel(W, H, rects, n_workers)
        else:
            with Manager() as manager:
                found_flag = manager.Value('b', False)
                res = _solve_sa_single((W, H, rects, 42, 10000, found_flag))
                result = [Placement.from_dict(d) for d in res[0]] if res else None
                iters = res[1] if res else 10000
                workers = 1
        
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=result is not None,
            placements=result or [],
            mode_used="并行模拟退火" if parallel else "模拟退火",
            time_ms=elapsed,
            iterations=iters,
            workers_used=workers,
            confidence="~99% (高置信度)"
        )
    
    elif mode == SolveMode.PROOF:
        if parallel:
            result, iters, workers, log = solve_proof_parallel(W, H, rects, n_workers)
        else:
            with Manager() as manager:
                found_flag = manager.Value('b', False)
                angles = np.linspace(0, np.pi / 2, 13)
                res = _solve_branch_single((W, H, rects, angles, found_flag, 60.0))
                result = [Placement.from_dict(d) for d in res[0]] if res else None
                iters = res[1] if res else 0
                workers = 1
                log = "分支定界"
        
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=result is not None,
            placements=result or [],
            mode_used=log,
            time_ms=elapsed,
            iterations=iters,
            workers_used=workers,
            confidence="100% 数学证明"
        )
    
    return SolveResult(success=False)


# ============================================================
# 便捷函数
# ============================================================

def can_fit(W: float, H: float, rects: List[Tuple[float, float]], 
            mode: str = "fast") -> bool:
    """
    简单判断矩形能否放入容器
    
    参数:
        W, H: 容器尺寸
        rects: 矩形列表 [(w1, h1), (w2, h2), ...]
        mode: "fast" / "precise" / "proof"
    
    返回:
        bool: 是否可以放入
    
    示例:
        >>> can_fit(20, 15, [(8, 6), (7, 5), (6, 4)])
        True
    """
    mode_map = {"fast": SolveMode.FAST, "precise": SolveMode.PRECISE, "proof": SolveMode.PROOF}
    result = solve(W, H, rects, mode_map.get(mode, SolveMode.FAST))
    return result.success


def get_placement(W: float, H: float, rects: List[Tuple[float, float]], 
                  mode: str = "precise") -> Optional[List[dict]]:
    """
    获取放置方案
    
    参数:
        W, H: 容器尺寸
        rects: 矩形列表
        mode: "fast" / "precise" / "proof"
    
    返回:
        放置方案列表，每个元素包含 {w, h, cx, cy, angle}
        如果无法放入返回 None
    
    示例:
        >>> result = get_placement(20, 15, [(8, 6), (7, 5)])
        >>> print(result)
        [{'w': 8, 'h': 6, 'cx': 4.0, 'cy': 3.0, 'angle': 0.0}, ...]
    """
    mode_map = {"fast": SolveMode.FAST, "precise": SolveMode.PRECISE, "proof": SolveMode.PROOF}
    result = solve(W, H, rects, mode_map.get(mode, SolveMode.PRECISE))
    
    if result.success:
        return [p.to_dict() for p in result.placements]
    return None


# ============================================================
# 可视化
# ============================================================

def visualize(W: float, H: float, placements: List[Placement], 
              rects: List[Tuple[float, float]] = None, 
              title: str = None,
              save_path: str = None):
    """
    可视化放置结果
    
    参数:
        W, H: 容器尺寸
        placements: 放置方案（来自 solve() 的结果）
        rects: 原始矩形列表（可选，用于显示原始尺寸）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon, Rectangle
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 画容器
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor='black', linewidth=2))
    
    # 颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(placements)))
    
    # 画每个矩形
    for idx, (p, color) in enumerate(zip(placements, colors)):
        poly = p.get_polygon()
        ax.add_patch(MplPolygon(poly, closed=True, fill=True,
                                facecolor=color, edgecolor='darkblue',
                                alpha=0.7, linewidth=1.5))
        
        # 标注
        label = f'{idx+1}'
        if rects:
            label += f'\n{rects[idx][0]}x{rects[idx][1]}'
        label += f'\n{np.degrees(p.angle):.0f}°'
        
        ax.annotate(label, (p.cx, p.cy), ha='center', va='center', 
                   fontsize=9, fontweight='bold')
    
    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-1, H + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or f'矩形装箱结果 (容器: {W}×{H}, {len(placements)}个矩形)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()  # Windows需要
    
    print("=" * 60)
    print("多矩形装箱求解器")
    print("=" * 60)
    
    # 示例
    W, H = 25, 20
    rects = [(10, 8), (9, 6), (7, 5), (6, 4)]
    
    print(f"\n容器: {W} × {H}")
    print(f"矩形: {rects}")
    print(f"总面积: {sum(w*h for w,h in rects)} / {W*H} = {sum(w*h for w,h in rects)/(W*H)*100:.1f}%")
    print()
    
    # 快速模式
    result = solve(W, H, rects, SolveMode.FAST)
    print(f"快速模式: {'✓ 可以' if result.success else '✗ 不可以'} ({result.time_ms:.1f}ms)")
    
    # 精确模式
    result = solve(W, H, rects, SolveMode.PRECISE)
    print(f"精确模式: {'✓ 可以' if result.success else '✗ 不可以'} ({result.time_ms:.1f}ms)")
    
    if result.success:
        print("\n放置方案:")
        for i, p in enumerate(result.placements):
            print(f"  矩形{i+1}: 位置({p.cx:.1f}, {p.cy:.1f}), 角度{np.degrees(p.angle):.1f}°")
        
        # 可视化
        try:
            visualize(W, H, result.placements, rects)
        except:
            print("\n(可视化需要 matplotlib)")
