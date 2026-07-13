"""Layer 1 검증: 충돌모델을 차체(범퍼)에서 바퀴(축거) 기준으로 바꿔 Hybrid A*를
재실행한다. 사용자 확정 사실(2026-07-08): 노면 경계에는 연석이 있어 바퀴는
못 넘지만 차체 오버행(앞 2.77m/뒤 2.72m)은 그 위로 넘어가도 된다.

핵심 가설: 바퀴 기준이면 U턴 필요폭이 25m->21m로 줄어(개방구역 실폭 21.9m 안에
들어감) 방향전환이 대폭 줄어든 깔끔한 회차가 나온다.

- 충돌 footprint = (앞 L=6.29m[앞축], 뒤 0m[뒷축], 반폭 1.2475m[전폭 보수적]).
  범퍼/뒷오버행은 연석 위로 넘어가도 되므로 충돌검사에서 제외.
- 기존 검증된 geometry/reference_path.json은 건드리지 않고 별도 파일로 저장.
- 검증도 바퀴 footprint로 수행하고, 참고로 "범퍼가 경계를 넘는 노드 수"(구
  바디 모델이 금지했던 것)도 함께 센다.
"""
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState
from planning.hybrid_astar import HybridAStarPlanner

SIM_DIR = Path(__file__).resolve().parent.parent


def _rect(x, y, theta, front_oh, rear_oh, half_w):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front_oh, half_w), (front_oh, -half_w), (-rear_oh, -half_w), (-rear_oh, half_w)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])


def main():
    env = BusHairpinEnv()

    # 바퀴 기준 footprint: 앞=앞축(L), 뒤=뒷축(0), 폭=전폭(보수적).
    wheel_fp = (env.model.L, 0.0, env.half_width)
    # 참고용 바디(구모델) footprint.
    body_fp = (env.front_oh, env.rear_oh, env.half_width)
    print(f"wheel footprint(front={wheel_fp[0]:.2f}, rear={wheel_fp[1]:.2f}, half_w={wheel_fp[2]:.3f})")
    print(f"body  footprint(front={body_fp[0]:.2f}, rear={body_fp[1]:.2f}, half_w={body_fp[2]:.3f})")

    fine_steer = tuple(round(-1.0 + 0.1 * i, 2) for i in range(21))
    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon,
        footprint=wheel_fp,
        delta_max=env.model.delta_max,
        arc_len=0.3,
        steer_fracs=fine_steer,
        xy_res=0.2,
        theta_res_deg=2.0,
        reverse_penalty=1.3,
        gear_switch_penalty=3.0,   # 거리 우선(사용자 선택): 전환을 과도히 억누르지 않는 기존값 유지
        tol_pos=0.6,
        tol_heading_deg=12.0,
        obstacle_polygons=env.obstacle_polygons,
        debug=True,
    )

    start = (env.start_x, env.start_y, env.start_heading)
    goal_rear_x = env.goal_x - env.d_ref * math.cos(env.goal_heading)
    goal_rear_y = env.goal_y - env.d_ref * math.sin(env.goal_heading)
    goal = (goal_rear_x, goal_rear_y, env.goal_heading)

    print("start", start, "goal", goal, flush=True)
    t0 = time.time()
    path = planner.plan(start, goal, L=env.model.L, max_iter=400000, bidirectional=True)
    print("elapsed", round(time.time() - t0, 1), "s", flush=True)

    if path is None:
        print("NO PATH FOUND (바퀴모델에서도 실패 — 해상도를 더 높여 재시도 필요)")
        return

    print(f"path nodes: {len(path)}")

    wheel_viol = 0
    body_viol = 0   # 범퍼가 연석을 넘는 노드 수 = 바퀴모델이 '풀어준' 양
    for (x, y, theta, gear) in path:
        if not env.boundary_polygon.contains(_rect(x, y, theta, *wheel_fp)):
            wheel_viol += 1
        if not env.boundary_polygon.contains(_rect(x, y, theta, *body_fp)):
            body_viol += 1
    print(f"wheel-model violations (should be 0): {wheel_viol}")
    print(f"body-model violations (범퍼가 연석 넘는 노드 = 구모델이 금지했던 것): {body_viol}")

    gear_changes = sum(1 for i in range(1, len(path)) if path[i][3] != path[i - 1][3] and path[i][3] != "START")
    total_len = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
    print(f"gear changes: {gear_changes}  total path length: {round(total_len, 1)} m", flush=True)

    waypoints, gears, rear_waypoints = [], [], []
    for (x, y, theta, gear) in path:
        fc = env.front_center(BicycleState(x=x, y=y, theta=theta, v=0.0, delta=0.0))
        waypoints.append([round(fc[0], 3), round(fc[1], 3)])
        gears.append(gear if gear != "START" else "FWD")
        rear_waypoints.append([round(x, 3), round(y, 3), round(theta, 6)])

    out = {"waypoints": waypoints, "gears": gears, "rear_waypoints": rear_waypoints,
           "collision_model": "wheel", "footprint": list(wheel_fp)}
    out_path = SIM_DIR / "geometry" / "reference_path_wheelmodel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}", flush=True)

    # 시각화: 바퀴 사각형(파랑/빨강 경로) + 범퍼가 넘는 구간 표시
    fig, ax = plt.subplots(figsize=(12, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)
    for i in range(1, len(rear_waypoints)):
        color = "tab:red" if gears[i] == "REV" else "tab:blue"
        ax.plot([rear_waypoints[i - 1][0], rear_waypoints[i][0]],
                [rear_waypoints[i - 1][1], rear_waypoints[i][1]], color=color, linewidth=1.5)
    # 몇 개 자세에서 바퀴 사각형 실루엣
    step = max(1, len(rear_waypoints) // 40)
    for i in range(0, len(rear_waypoints), step):
        x, y, th = rear_waypoints[i]
        wx, wy = _rect(x, y, th, *wheel_fp).exterior.xy
        ax.plot(wx, wy, color="0.6", linewidth=0.4)
    ax.plot([], [], color="tab:blue", linewidth=2, label="FWD")
    ax.plot([], [], color="tab:red", linewidth=2, label="REV")
    ax.plot(rear_waypoints[0][0], rear_waypoints[0][1], "g^", markersize=12, label="start (rear axle)")
    ax.plot(env.goal_x, env.goal_y, "kx", markersize=14, mew=2, label="goal (front-center)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"WHEEL-model Hybrid A*: gear_changes={gear_changes}, len={total_len:.1f}m, "
                 f"wheel_viol={wheel_viol}, bumper-over-curb nodes={body_viol}")
    fig.tight_layout()
    out_png = SIM_DIR / "outputs" / "hybrid_astar_wheelmodel.png"
    fig.savefig(out_png, dpi=140)
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
