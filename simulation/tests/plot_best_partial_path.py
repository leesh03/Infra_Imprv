"""사용자 요청: '현재 나온 경로'를 PNG로 확인. 주의 — Hybrid A*는 아직
goal을 정확히 못 찾았다(NO PATH FOUND, 방향오차 150도대). 여기서 그리는
건 완성된 경로가 아니라 **정방향 탐색이 도달한 가장 가까운 노드까지의
부분 경로**(위치+방향 결합점수 기준 최선)다 — 실제로 어디까지 갔고 어디서
막혔는지 보여주는 진단용 그림. FWD/REV 구간을 색으로 구분."""
import math

import matplotlib.pyplot as plt

from envs.bus_hairpin_env import BusHairpinEnv
from planning.hybrid_astar import HybridAStarPlanner


def main():
    env = BusHairpinEnv()
    footprint = (env.front_oh, env.rear_oh, env.half_width)

    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon,
        footprint=footprint,
        delta_max=env.model.delta_max,
        arc_len=0.6,
        steer_fracs=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0, -0.1, -0.25, -0.4, -0.55, -0.7, -0.85, -1.0),
        xy_res=0.35,
        theta_res_deg=4.0,
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

    # 사용자 지정: goal 도착은 반드시 FWD(앞면이 닿는 전진)여야 하고, 마지막
    # 한 스텝만이 아니라 최근 3스텝 연속 FWD여야 한다(REV로 목표를 지나쳤다가
    # 살짝 되돌아오는 눈속임 방지).
    node, closed_a, counter, diag = planner._expand(
        start, goal, env.model.L, 400000, label="fwd",
        arrival_gear_required="FWD", arrival_min_fwd_run=3,
    )
    if node is not None:
        path = planner._reconstruct(node)
        complete = True
    else:
        best_node = diag["best_combined_valid_node"] or diag["best_combined_node"]
        path = planner._reconstruct(best_node)
        complete = False
        valid = diag["best_combined_valid_node"] is not None
        print(f"NO PATH FOUND -- plotting BEST {'VALID(FWD-arrival) ' if valid else '(NO FWD-valid candidate, showing gear-agnostic) '}"
              f"PARTIAL path (dpos={math.hypot(best_node.x-goal[0], best_node.y-goal[1]):.3f}m, gear={best_node.gear})")

    # front-center 웨이포인트 + gear (FWD/REV 색상 구분용)
    segs_fwd = []
    segs_rev = []
    for i in range(1, len(path)):
        x0, y0, th0, _ = path[i - 1]
        x1, y1, th1, gear = path[i]
        seg = ((x0, y0), (x1, y1))
        if gear == "REV":
            segs_rev.append(seg)
        else:
            segs_fwd.append(seg)

    fig, ax = plt.subplots(figsize=(11, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)

    for (p0, p1) in segs_fwd:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:blue", linewidth=2.5)
    for (p0, p1) in segs_rev:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:red", linewidth=2.5)
    # 범례용 더미
    ax.plot([], [], color="tab:blue", linewidth=2.5, label="FWD (forward)")
    ax.plot([], [], color="tab:red", linewidth=2.5, label="REV (reverse)")

    ax.plot(start[0], start[1], "g^", markersize=12, label="start (rear axle)")
    ax.plot(goal[0], goal[1], "m*", markersize=16, label="goal (rear axle)")
    last = path[-1]
    ax.plot(last[0], last[1], "o", color="darkorange", markersize=10,
            label=f"path end (dpos={math.hypot(last[0]-goal[0], last[1]-goal[1]):.2f}m, "
                  f"dhead={math.degrees(abs(((last[2]-goal[2]+math.pi)%(2*math.pi))-math.pi)):.1f}deg)")

    title = "Hybrid A* result: " + ("COMPLETE PATH" if complete else "BEST PARTIAL PATH (goal NOT reached)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    out_path = "outputs/best_partial_path.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}  (complete={complete}, fwd_segs={len(segs_fwd)}, rev_segs={len(segs_rev)})")


if __name__ == "__main__":
    main()
