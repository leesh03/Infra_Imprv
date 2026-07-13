"""헤어핀 팁 밀집 구간(런 5~25, 20회 방향전환, 1.6m x 2.6m 안에 몰려있음)을
통째로 자유 기어 CMA-ES로 재최적화 — 인접 두 런/고정 기어만 보던
try_connect보다 훨씬 넓은 재구조화가 가능한지 확인한다."""
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

CLUSTER_RUN_START = 5   # runs[5]..runs[25] (0-indexed) -- 노드 193..267
CLUSTER_RUN_END = 25


def main():
    env = BusHairpinEnv()
    with open(SIM_DIR / "geometry" / "reference_path_repaired.json", encoding="utf-8") as f:
        path_json = json.load(f)  # 이미 26회로 줄인 결과 위에 이어서 시도

    rw = path_json["rear_waypoints"]
    gears_orig = path_json["gears"]
    runs = path_json_to_runs(path_json)

    # reference_path_repaired.json은 스플라이스로 노드 수가 바뀌었으므로,
    # 원본 run 인덱스(5~25)가 아니라 "밀집 구간"을 x,y 범위로 다시 찾는다.
    cluster_runs = [i for i, (s, e, g) in enumerate(runs) if e - s <= 10 and 15 < rw[s][0] < 19]
    if not cluster_runs:
        print("밀집 구간을 못 찾음 — 종료")
        return
    lo, hi = min(cluster_runs), max(cluster_runs)
    # 양옆으로 "짧은" 런(<=15노드)이 이어지면만 포함 — 무작정 +-1하면 바로
    # 옆의 130노드짜리 진출 램프까지 통째로 끌려들어온다(실측으로 확인한 버그).
    while lo > 0 and (runs[lo - 1][1] - runs[lo - 1][0]) <= 15:
        lo -= 1
    while hi < len(runs) - 1 and (runs[hi + 1][1] - runs[hi + 1][0]) <= 15:
        hi += 1
    run_a, run_b = runs[lo], runs[hi]
    entry_pose = tuple(rw[run_a[0]])
    exit_pose = tuple(rw[run_b[1]])
    n_switch_orig = hi - lo  # runs[lo..hi] 사이 스위치 수

    print(f"클러스터: run[{lo}..{hi}] node[{run_a[0]}..{run_b[1]}] "
          f"원본 스위치수={n_switch_orig} entry={entry_pose} exit={exit_pose}")

    footprint = (env.front_oh, env.rear_oh, env.half_width)
    kappa_max = 1.0 / env.model.R

    t0 = time.time()
    result = search_cluster(
        entry_pose, exit_pose, kappa_max, footprint, env.boundary_polygon,
        K_list=[16, 12, 9, 7], len_max=12.0, ds=0.12, n_random=400000,
        n_polish_restarts=3, polish_maxiter=200,
        n_jobs=7, w_switch=3.0, seed0=42,
    )
    elapsed = time.time() - t0
    print(f"\nelapsed {elapsed:.1f}s")

    if result is None:
        print(f"실패 — 이 창(원본 스위치수={n_switch_orig})에서 재구조화 못 찾음. 원본 유지.")
        return

    n_switch_new, dense, K_used = result
    print(f"성공: K={K_used} n_switch={n_switch_new} (원본 {n_switch_orig} -> {n_switch_new})")

    if n_switch_new >= n_switch_orig:
        print("개선 없음(스위치 수가 같거나 더 많음) — 적용하지 않음.")
        return

    # 스플라이스
    new_rw = [list(p) for p in rw[:run_a[0]]]
    new_gears = gears_orig[:run_a[0]] if run_a[0] > 0 else []
    if not new_rw:
        new_rw.append([dense[0][0], dense[0][1], dense[0][2]])
        new_gears.append("FWD")
    for (x, y, th, gear) in dense[1:]:
        new_rw.append([x, y, th])
        new_gears.append(gear)
    for idx in range(run_b[1] + 1, len(rw)):
        new_rw.append(list(rw[idx]))
        new_gears.append(gears_orig[idx])

    violations = 0
    for (x, y, th) in new_rw:
        st = BicycleState(x=x, y=y, theta=th, v=0.0, delta=0.0)
        if not env.boundary_polygon.contains(env._bus_polygon(st)):
            violations += 1
    print(f"전체 재검증 violations (should be 0): {violations}")
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
    out_path = SIM_DIR / "geometry" / "reference_path_repaired2.json"
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
    ax.set_title(f"Cluster-repaired: switches {n_switch_total_orig}->{n_switch_total_new}, "
                 f"len={total_len_new:.1f}m, violations={violations}")
    fig.tight_layout()
    out_png = SIM_DIR / "outputs" / "reference_path_repaired2.png"
    fig.savefig(out_png, dpi=140)
    print(f"저장: {out_png}")

    # 클러스터 확대 뷰도 저장(변화가 실제로 보이도록)
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    ax2.plot(px, py, "k-", linewidth=1)
    for i in range(1, len(waypoints)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax2.plot([waypoints[i-1][0], waypoints[i][0]], [waypoints[i-1][1], waypoints[i][1]],
                  color=color, linewidth=2.0)
    cx, cy = entry_pose[0], entry_pose[1]
    ax2.set_xlim(cx - 6, cx + 6)
    ax2.set_ylim(cy - 6, cy + 6)
    ax2.set_aspect("equal")
    ax2.set_title(f"Cluster zoom: switches {n_switch_orig}->{n_switch_new} in this window")
    fig2.tight_layout()
    out_png2 = SIM_DIR / "outputs" / "reference_path_repaired2_zoom.png"
    fig2.savefig(out_png2, dpi=140)
    print(f"저장: {out_png2}")


if __name__ == "__main__":
    main()
