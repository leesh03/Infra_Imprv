"""사용자가 제시한 실제 기사님 주행 예시(넓게 스윙해서 정점 근처에서 한두 번만
방향전환)를 참고해, 루프 전체(진입 지점 ~ 진출 지점)를 통째로 다시 찾는다.

지금까지의 국소 수정(try_connect, search_cluster 좁은 창)은 전부 기존
hybrid A*의 "한쪽 벽에 바짝 붙어 20회+ 방향전환하는" 좁은 경로의 정확한
좌표를 시작/끝점(anchor)으로 고정했다 — 그래서 그 안에서 아무리 자유롭게
탐색해도 근본적으로 다른(넓게 스윙하는) 전략은 애초에 나올 수 없는
구조였다. 이번엔 진입/진출 지점을 훨씬 더 이르게/늦게 잡아(원본 경로가
처음으로 후진하는 지점 ~ 마지막으로 전진을 시작하는 지점) 루프 전체 구간에
자유도를 주고, len_max도 크게 줘서(넓은 원호 허용) 완전히 다른 모양을
찾을 수 있는지 확인한다."""
import json
import math
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState, front_center
from planning.switch_repair import path_json_to_runs, search_cluster

SIM_DIR = Path(__file__).resolve().parent.parent


def main():
    env = BusHairpinEnv()
    with open(SIM_DIR / "geometry" / "reference_path.json", encoding="utf-8") as f:
        path_json = json.load(f)

    rw = path_json["rear_waypoints"]
    gears_orig = path_json["gears"]
    runs = path_json_to_runs(path_json)

    # run0(FWD,131) 끝 = 처음 후진하는 지점(루프 진입), run26(FWD,130) 시작 =
    # 마지막으로 쭉 전진을 시작하는 지점(루프 진출). 그 사이 전체(런1~25,
    # 25회 방향전환)를 통째로 다시 찾는다.
    entry_run_idx = 0
    exit_run_idx = next(i for i, (s, e, g) in enumerate(runs) if e - s > 100 and i > 0)
    run_a, run_b = runs[entry_run_idx], runs[exit_run_idx]
    entry_pose = tuple(rw[run_a[1]])
    exit_pose = tuple(rw[run_b[0]])
    n_switch_orig = exit_run_idx - entry_run_idx - 1

    print(f"루프 전체 창: node[{run_a[1]}..{run_b[0]}] 원본 스위치수={n_switch_orig}")
    print(f"entry={entry_pose} exit={exit_pose}")
    dx = exit_pose[0] - entry_pose[0]
    dy = exit_pose[1] - entry_pose[1]
    print(f"직선거리={math.hypot(dx,dy):.1f}m")

    # 충돌검사 footprint를 차체 범퍼(front_oh=d_ref=9.06m, 앞 오버행 포함)가
    # 아니라 바퀴 위치(축거 L=6.29m, 뒤는 뒷축=0) 기준으로 바꾼다 — 경계
    # 폴리곤이 DXF의 "밑 도면"(포장 평면도) 레이어에서 온 걸로 봐서 벽이
    # 아니라 노면 가장자리이므로, 실제 제약은 "바퀴가 노면 안에 있어야
    # 한다"이지 "범퍼가 안 넘어야 한다"가 아니다(사용자 지적). 폭은
    # 보수적으로 차체 전폭(뒷 윤거 실측치가 없어 안전 마진으로 그대로 둠).
    footprint = (env.model.L, 0.0, env.half_width)
    kappa_max = 1.0 / env.model.R

    t0 = time.time()
    result = search_cluster(
        entry_pose, exit_pose, kappa_max, footprint, env.boundary_polygon,
        K_list=[8, 5, 4, 3], len_max=25.0, ds=0.15, n_random=30000,
        n_polish_restarts=3, polish_maxiter=250,
        n_jobs=7, w_switch=5.0, seed0=123,
    )
    elapsed = time.time() - t0
    print(f"\nelapsed {elapsed:.1f}s")

    if result is None:
        print(f"실패 — 루프 전체 재구조화 못 찾음(원본 스위치수={n_switch_orig}). 원본 유지.")
        return

    n_switch_new, dense, K_used = result
    print(f"성공: K={K_used} n_switch={n_switch_new} (원본 {n_switch_orig} -> {n_switch_new})")

    # 스플라이스
    new_rw = [list(p) for p in rw[:run_a[1]]]
    new_gears = gears_orig[:run_a[1]] if run_a[1] > 0 else []
    if not new_rw:
        new_rw.append([dense[0][0], dense[0][1], dense[0][2]])
        new_gears.append("FWD")
    for (x, y, th, gear) in dense[1:]:
        new_rw.append([x, y, th])
        new_gears.append(gear)
    for idx in range(run_b[0] + 1, len(rw)):
        new_rw.append(list(rw[idx]))
        new_gears.append(gears_orig[idx])

    # env._bus_polygon은 범퍼 기준(구모델)이므로 안 쓴다 — 새 footprint(바퀴
    # 기준, 위에서 정의)로 직접 재검증한다.
    from planning.evolutionary import bus_polygon as _wheel_bus_polygon
    front_oh_w, rear_oh_w, half_w = footprint
    violations = 0
    for (x, y, th) in new_rw:
        poly = _wheel_bus_polygon(x, y, th, front_oh_w, rear_oh_w, half_w)
        if not env.boundary_polygon.contains(poly):
            violations += 1
    print(f"전체 재검증 violations (should be 0, 바퀴기준 footprint): {violations}")
    if violations > 0:
        print("경고: 위반 발견 — 저장하지 않음")
        return

    n_switch_total_new = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    n_switch_total_orig = sum(1 for i in range(1, len(gears_orig)) if gears_orig[i] != gears_orig[i - 1])
    total_len_new = sum(
        math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
        for i in range(1, len(new_rw))
    )
    print(f"전체 스위치: {n_switch_total_orig} -> {n_switch_total_new}")
    fc = front_center(*new_rw[-1], env.d_ref)
    dpos = math.hypot(env.goal_x - fc[0], env.goal_y - fc[1])
    print(f"goal dpos: {dpos:.4f}m, final gear: {new_gears[-1]}")

    waypoints = [[round(x, 3), round(y, 3)] for x, y in
                 (front_center(*p, env.d_ref) for p in new_rw)]
    out = {"waypoints": waypoints, "gears": new_gears,
           "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw]}
    out_path = SIM_DIR / "geometry" / "reference_path_wideloop.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")

    fig, ax = plt.subplots(figsize=(11, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)
    for i in range(1, len(waypoints)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax.plot([waypoints[i-1][0], waypoints[i][0]], [waypoints[i-1][1], waypoints[i][1]],
                 color=color, linewidth=2.0)
    ax.plot([], [], color="tab:blue", linewidth=2.5, label="FWD")
    ax.plot([], [], color="tab:red", linewidth=2.5, label="REV")
    ax.plot(waypoints[0][0], waypoints[0][1], "g^", markersize=12, label="start")
    ax.plot(waypoints[-1][0], waypoints[-1][1], "m*", markersize=16, label="end")
    ax.plot(env.goal_x, env.goal_y, "kx", markersize=14, mew=2, label="goal")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Wide-loop repair: switches {n_switch_total_orig}->{n_switch_total_new}, "
                 f"len={total_len_new:.1f}m, violations={violations}")
    fig.tight_layout()
    out_png = SIM_DIR / "outputs" / "reference_path_wideloop.png"
    fig.savefig(out_png, dpi=140)
    print(f"저장: {out_png}")


if __name__ == "__main__":
    main()
