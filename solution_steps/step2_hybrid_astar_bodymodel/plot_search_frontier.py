"""사용자 요청(RL_NEXT_STEPS.md PHASE A 후속): 정방향(start 기준)/역방향(goal
기준) Hybrid A* 닫힌 노드 집합이 실제 도로 위 어디에 분포하는지 시각화한다.
start/goal 포즈를 어느 방향으로 조정해야 할지 판단하기 위한 진단용 스크립트
— 학습/최종 산출물과 무관."""
import math
import time

import matplotlib.pyplot as plt

from envs.bus_hairpin_env import BusHairpinEnv
from planning.hybrid_astar import HybridAStarPlanner

SIM_DIR_OUTPUTS = "outputs/search_frontier.png"


def main():
    env = BusHairpinEnv()
    footprint = (env.front_oh, env.rear_oh, env.half_width)

    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon,
        footprint=footprint,
        delta_max=env.model.delta_max,
        arc_len=1.0,
        steer_fracs=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0),
        xy_res=0.5,
        theta_res_deg=6.0,
        reverse_penalty=1.1,
        gear_switch_penalty=0.5,
        tol_pos=0.6,
        tol_heading_deg=12.0,
        obstacle_polygons=env.obstacle_polygons,
        debug=True,
    )

    start = (env.start_x, env.start_y, env.start_heading)
    goal_rear_x = env.goal_x - env.d_ref * math.cos(env.goal_heading)
    goal_rear_y = env.goal_y - env.d_ref * math.sin(env.goal_heading)
    goal = (goal_rear_x, goal_rear_y, env.goal_heading)

    t0 = time.time()
    node_a, closed_a, counter_a, diag_a = planner._expand(start, goal, env.model.L, 400000, label="fwd")
    node_b, closed_b, counter_b, diag_b = planner._expand(goal, start, env.model.L, 400000, label="bwd")
    print("elapsed", time.time() - t0, "s")

    # 위치는 겹치는데 (x,y,theta) 셀 교집합이 0인 게 이산화 우연인지, 정말
    # 방향이 구조적으로 안 맞는 것인지 정량 확인: A의 각 노드에 대해 위치가
    # 가까운(0.5m 이내) B 노드들 중 heading 차이가 가장 작은 값을 구해 분포를
    # 본다.
    import random
    random.seed(0)
    a_list = list(closed_a.values())
    b_list = list(closed_b.values())
    from collections import defaultdict
    bucket = defaultdict(list)
    POS_BIN = 1.0
    for n in b_list:
        bucket[(round(n.x / POS_BIN), round(n.y / POS_BIN))].append(n)

    sample = random.sample(a_list, min(1500, len(a_list)))
    gaps = []
    for n in sample:
        bx0, by0 = round(n.x / POS_BIN), round(n.y / POS_BIN)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cand in bucket.get((bx0 + dx, by0 + dy), []):
                    if math.hypot(cand.x - n.x, cand.y - n.y) <= 0.6:
                        dth = abs(((cand.theta - n.theta + math.pi) % (2 * math.pi)) - math.pi)
                        if best is None or dth < best:
                            best = dth
        if best is not None:
            gaps.append(math.degrees(best))
    if gaps:
        gaps.sort()
        n = len(gaps)
        print(f"[heading-gap @ matching position] n_samples_with_neighbor={n}/{len(sample)}")
        print(f"  min={gaps[0]:.1f} deg  median={gaps[n//2]:.1f} deg  "
              f"p90={gaps[int(n*0.9)]:.1f} deg  max={gaps[-1]:.1f} deg")
    else:
        print("[heading-gap] no A-sample had any B neighbor within 0.6m — positions themselves don't overlap that closely")

    fig, ax = plt.subplots(figsize=(10, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)

    ax_pts = [(n.x, n.y) for n in closed_a.values()]
    bx_pts = [(n.x, n.y) for n in closed_b.values()]
    if ax_pts:
        axs, ays = zip(*ax_pts)
        ax.scatter(axs, ays, s=4, color="tab:blue", alpha=0.4, label=f"forward closed (from start) n={len(ax_pts)}")
    if bx_pts:
        bxs, bys = zip(*bx_pts)
        ax.scatter(bxs, bys, s=4, color="tab:red", alpha=0.4, label=f"backward closed (from goal) n={len(bx_pts)}")

    ax.plot(start[0], start[1], "g^", markersize=14, label="start (rear axle)")
    ax.plot(goal[0], goal[1], "m*", markersize=16, label="goal (rear axle)")

    bn_a = diag_a["best_dpos_node"] if diag_a else None
    bn_b = diag_b["best_dpos_node"] if diag_b else None
    if bn_a:
        ax.plot(bn_a.x, bn_a.y, "o", color="navy", markersize=10, label="forward best_dpos node")
    if bn_b:
        ax.plot(bn_b.x, bn_b.y, "o", color="darkred", markersize=10, label="backward best_dpos node")

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Hybrid A* search frontier (forward vs backward closed sets)")
    fig.tight_layout()
    fig.savefig(SIM_DIR_OUTPUTS, dpi=140)
    print(f"saved {SIM_DIR_OUTPUTS}")


if __name__ == "__main__":
    main()
