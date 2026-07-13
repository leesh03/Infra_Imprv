"""run_switch_repair.py가 찾은 제거 가능한 방향전환들을 실제로 원본
reference_path.json에 스플라이스해 넣고, 결과 전체를 원본 정밀 폴리곤으로
재검증한다. 탐색 자체(각 스위치의 try_connect)는 비용이 커서 이미 성공이
확인된 스위치 인덱스만 다시 계산한다(같은 seed 공식이라 재현됨)."""
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
from planning.switch_repair import path_json_to_runs, try_connect

SIM_DIR = Path(__file__).resolve().parent.parent

# run_switch_repair.py --maxiter 200 --n-restarts 3 실행에서 확인된 제거
# 가능한 스위치 인덱스(0-based, results[i] <-> runs[i]<->runs[i+1])와 기어.
SUCCESSFUL_SWITCHES = [
    (7, "REV"),
    (9, "FWD"),
    (14, "FWD"),
]
MAXITER = 200
N_RESTARTS = 3
K_LOCAL = 3
LEN_CAP = 15.0


def main():
    env = BusHairpinEnv()
    with open(SIM_DIR / "geometry" / "reference_path.json", encoding="utf-8") as f:
        path_json = json.load(f)

    rw = path_json["rear_waypoints"]
    gears_orig = path_json["gears"]
    runs = path_json_to_runs(path_json)
    footprint = (env.front_oh, env.rear_oh, env.half_width)
    kappa_max = 1.0 / env.model.R

    from shapely.prepared import prep
    fast_boundary = env.boundary_polygon.simplify(0.05, preserve_topology=True).buffer(-0.05)
    prep_fast = prep(fast_boundary)
    prep_precise = prep(env.boundary_polygon)

    splices = []  # (run_a_start, run_b_end, gear_label, dense)
    for i, gear_label in SUCCESSFUL_SWITCHES:
        run_a, run_b = runs[i], runs[i + 1]
        entry_pose = tuple(rw[run_a[0]])
        exit_pose = tuple(rw[run_b[1]])
        gear_sign = 1.0 if gear_label == "FWD" else -1.0
        t0 = time.time()
        dense = try_connect(
            entry_pose, exit_pose, gear_sign, kappa_max, footprint,
            fast_boundary, prep_fast, env.boundary_polygon, prep_precise,
            k_local=K_LOCAL, len_cap=LEN_CAP, maxiter=MAXITER, n_restarts=N_RESTARTS, seed0=i * 17,
        )
        print(f"switch {i+1}: gear={gear_label} recompute {'OK' if dense else 'FAILED'} "
              f"({time.time()-t0:.1f}s, {len(dense) if dense else 0} pts)")
        if dense is None:
            print(f"  경고: switch {i+1} 재현 실패 — 이번 스플라이스에서 제외")
            continue
        splices.append((run_a[0], run_b[1], gear_label, dense))

    if not splices:
        print("적용 가능한 스플라이스 없음 — 종료")
        return

    splices.sort(key=lambda s: s[0])
    new_rw = []
    new_gears = ["FWD"]  # gears[0]은 원본처럼 더미
    node_i = 0
    n = len(rw)
    splice_idx = 0
    while node_i < n:
        if splice_idx < len(splices) and splices[splice_idx][0] == node_i:
            start, end, gear_label, dense = splices[splice_idx]
            if not new_rw:
                new_rw.append(list(dense[0]))
            for (x, y, th) in dense[1:]:
                new_rw.append([x, y, th])
                new_gears.append(gear_label)
            node_i = end + 1  # run_b의 끝 노드까지 건너뜀(그 노드는 dense의 마지막점으로 대체됨)
            splice_idx += 1
            continue
        if not new_rw:
            new_rw.append(list(rw[node_i]))
        else:
            new_rw.append(list(rw[node_i]))
            new_gears.append(gears_orig[node_i])
        node_i += 1

    print(f"\n원본 {n}노드 -> 새 경로 {len(new_rw)}노드")

    # 전체 재검증: 매 노드에서 버스 OBB가 원본 정밀 경계 폴리곤 안에 있는지
    violations = 0
    for (x, y, th) in new_rw:
        st = BicycleState(x=x, y=y, theta=th, v=0.0, delta=0.0)
        if not env.boundary_polygon.contains(env._bus_polygon(st)):
            violations += 1
    print(f"violations (should be 0): {violations}")

    n_switch_new = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    total_len_new = sum(
        math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
        for i in range(1, len(new_rw))
    )
    n_switch_orig = sum(1 for i in range(1, len(gears_orig)) if gears_orig[i] != gears_orig[i - 1])
    total_len_orig = sum(
        math.hypot(rw[i][0] - rw[i - 1][0], rw[i][1] - rw[i - 1][1])
        for i in range(1, len(rw))
    )
    print(f"gear switches: {n_switch_orig} -> {n_switch_new}")
    print(f"total length: {total_len_orig:.1f}m -> {total_len_new:.1f}m")

    # 도착 조건(전진 도착) 확인 — 꼬리는 안 건드렸으니 원본과 동일해야 함
    dx = env.goal_x - front_center(*new_rw[-1], env.d_ref)[0]
    fc = front_center(*new_rw[-1], env.d_ref)
    dpos = math.hypot(env.goal_x - fc[0], env.goal_y - fc[1])
    print(f"goal front-center dpos: {dpos:.4f}m (tol {env.tol_pos}m)")
    print(f"final gear: {new_gears[-1]}")

    if violations > 0:
        print("\n경고: 위반 발견 — 저장하지 않음")
        return

    waypoints = []
    for (x, y, th) in new_rw:
        fcx, fcy = front_center(x, y, th, env.d_ref)
        waypoints.append([round(fcx, 3), round(fcy, 3)])
    out = {
        "waypoints": waypoints,
        "gears": new_gears,
        "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw],
    }
    out_path = SIM_DIR / "geometry" / "reference_path_repaired.json"
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
    ax.set_title(
        f"Switch-repaired path: switches {n_switch_orig}->{n_switch_new}, "
        f"len {total_len_orig:.1f}m->{total_len_new:.1f}m, violations={violations}"
    )
    fig.tight_layout()
    out_png = SIM_DIR / "outputs" / "reference_path_repaired.png"
    fig.savefig(out_png, dpi=140)
    print(f"저장: {out_png}")


if __name__ == "__main__":
    main()
