"""Hybrid A*로 start -> goal 전/후진 경로를 탐색하고 결과를 검증 후
geometry/reference_path.json 으로 저장 (front-center 웨이포인트 + gear)."""
import json
import math
import time
from pathlib import Path

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState
from planning.hybrid_astar import HybridAStarPlanner

SIM_DIR = Path(__file__).resolve().parent.parent


def main():
    env = BusHairpinEnv()
    footprint = (env.front_oh, env.rear_oh, env.half_width)

    # RL_WHEEL_GEOMETRY_REPLAN.md PHASE 1: env.model.delta_max는 이미 R_min의
    # 측정기준(outer_front_tire_track)을 반영해 보정된 값 — 플래너에 그대로
    # 전달한다(플래너 내부에서 R_min으로부터 재계산하지 않음).
    # RL_NEXT_STEPS.md PHASE A 항목4: 양방향 탐색으로도 두 트리가 전혀 겹치지
    # 않아(교집합 0), 새로 보정된 delta_max 기준으로 해상도를 한 번 더
    # 좁혀본다 — 이전 스윕(§RL_PROGRESS_REPORT.md)은 보정 전 delta_max
    # 기준이었으므로 무효한 반복이 아니다. 이걸로도 안 풀리면 "기하학적 불가"
    # 판정으로 넘어간다(무한 스윕 금지).
    # 사용자 요청: 전진 단독이 아니라 "조정(전후진 전환)을 최소화하며 후진을
    # 허용"하는 경로를 원함 — 기어 자체는 원래부터 FWD/REV 둘 다 탐색 대상
    # 이었으나, gear_switch_penalty(3.0)가 다회 전환이 필요한 깊은 분기를
    # 비용으로 짓눌러 제대로 못 뻗어봤을 수 있어 낮춰서 재시도.
    # 2026-07-04: 시작/목표를 경사로 끝단(캡) 기준으로 재정의한 뒤 역방향
    # 트리(목표 쪽)가 108개 노드로 유독 작아 여기가 병목 — 프리미티브를 더
    # 세밀하게(arc_len 1.0->0.6, xy_res 0.5->0.35, theta_res 6->4deg) 해서
    # 목표 근처 좁은 구간에서 더 촘촘히 뻗어볼 수 있는지 확인.
    # 2026-07-05: 역방향 트리가 96노드에서 조기 고갈되는 문제를 진단한 결과,
    # 실제 기하학적 함정이 아니라 격자 해상도가 좁은 구간의 흔들기(shunting)
    # 동작을 표현하기엔 너무 거칠어서 생긴 이산화 artifact로 확인됨(arc_len
    # 0.3/xy_res 0.2/theta_res 2deg로는 30000pops까지도 open_heap이 죽지
    # 않고 계속 성장). 그래서 해상도를 한 단계 더 세밀하게 올려 재탐색.
    #
    # 2026-07-05(2차): 위 설정으로 경로는 찾았으나(388노드), 사용자가 결과
    # 그림을 보고 "말이 안 되는 경로"라고 지적 — 우측 코너를 불필요하게 크게
    # 돌고 전후진 전환이 과도(38회)했다. 원인은 gear_switch_penalty를 이전
    # 시행착오에서 "일단 아무 경로라도 찾자"는 목적으로 3.0->0.5로 낮춰둔
    # 채로 뒀기 때문 — 전환 비용이 스텝 비용(arc_len=0.3)의 약 1.7배에
    # 불과해 A*가 전환을 거의 공짜로 취급했다. 이제 경로 존재 자체는
    # 확인됐으니, 전환/후진에 다시 의미 있는 비용을 매겨 "최단거리 + 최소
    # 전후진 조정"을 사용자 요구대로 최적화한다.
    fine_steer = tuple(round(-1.0 + 0.1 * i, 2) for i in range(21))
    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon,
        footprint=footprint,
        delta_max=env.model.delta_max,
        arc_len=0.3,
        steer_fracs=fine_steer,
        xy_res=0.2,
        theta_res_deg=2.0,
        reverse_penalty=1.3,
        gear_switch_penalty=3.0,
        tol_pos=0.6,
        tol_heading_deg=12.0,
        obstacle_polygons=env.obstacle_polygons,
        debug=True,
    )

    start = (env.start_x, env.start_y, env.start_heading)
    # 목표는 앞면중심 기준이므로 뒷축 pose로 역산
    goal_rear_x = env.goal_x - env.d_ref * math.cos(env.goal_heading)
    goal_rear_y = env.goal_y - env.d_ref * math.sin(env.goal_heading)
    goal = (goal_rear_x, goal_rear_y, env.goal_heading)

    print("start", start, "goal", goal)
    t0 = time.time()
    # RL_NEXT_STEPS.md PHASE A 항목1: 단방향 탐색이 실패하면 goal에서 역방향
    # 트리를 추가로 키워 두 트리가 겹치는 지점에서 경로를 잇는다.
    path = planner.plan(start, goal, L=env.model.L, max_iter=400000, bidirectional=True)
    print("elapsed", time.time() - t0, "s")

    if path is None:
        print("NO PATH FOUND")
        return

    print(f"path nodes: {len(path)}")

    # 검증: 매 노드에서 버스 OBB가 경계 안에 있는지 재확인
    violations = 0
    for (x, y, theta, gear) in path:
        st = BicycleState(x=x, y=y, theta=theta, v=0.0, delta=0.0)
        if not env.boundary_polygon.contains(env._bus_polygon(st)):
            violations += 1
    print("violations (should be 0):", violations)

    gear_changes = sum(1 for i in range(1, len(path)) if path[i][3] != path[i - 1][3] and path[i][3] != "START")
    total_len = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
    print("gear changes:", gear_changes, "total path length:", round(total_len, 1))

    # front-center 웨이포인트로 변환해 저장
    waypoints = []
    gears = []
    for (x, y, theta, gear) in path:
        fc = env.front_center(BicycleState(x=x, y=y, theta=theta, v=0.0, delta=0.0))
        waypoints.append([round(fc[0], 3), round(fc[1], 3)])
        gears.append(gear if gear != "START" else "FWD")

    out = {"waypoints": waypoints, "gears": gears}
    with open(SIM_DIR / "geometry" / "reference_path.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved geometry/reference_path.json")


if __name__ == "__main__":
    main()
