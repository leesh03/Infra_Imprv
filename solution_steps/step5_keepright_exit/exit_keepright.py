"""우측통행 진출 (clean v2) — 윗 경사로가 2차선 양방향이므로, 마지막 전진
진출구간(run4)을 진행방향 오른쪽 차선으로 붙이고 오른쪽 차선에 도착한다.

- 회차(run0~run3, 2차 후진까지)는 그대로 둔다.
- run4를 진행방향 오른쪽(heading -90°)으로 옆이동. 단 곡선 구간(포켓 탈출)은
  원본 곡률로 보호(옆이동 0)하고 직진 구간만 민다. 전이는 스무딩.
- 끝점(도착)은 캡 중앙이 아니라 오른쪽 차선(캡 중앙 + 오른쪽 offset)에 둔다.
- R_min 곡률한계·바퀴 경계위반을 재검증.

기존 shift_exit_keepright.py / replan_exit_keepright.py(다른 도구가 수정)를
건드리지 않기 위해 별도 파일로 새로 작성.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

SIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_DIR))
from envs.bus_hairpin_env import BusHairpinEnv  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def split_runs(rw, gears):
    n = len(rw); edge = gears[1:n]; runs = []; k = 0; m = len(edge)
    while k < m:
        g = edge[k]; j = k
        while j < m and edge[j] == g:
            j += 1
        runs.append((k, j, g)); k = j
    return runs


def wheel_poly(x, y, th, wb, hw):
    c, sn = math.cos(th), math.sin(th)
    loc = [(wb, hw), (wb, -hw), (0.0, -hw), (0.0, hw)]
    return Polygon([(x + lx * c - ly * sn, y + lx * sn + ly * c) for lx, ly in loc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-right", type=float, default=2.2,
                    help="오른쪽 차선 이동량[m] (좌측hug->중앙 ~1.6m 이후 우측차선까지)")
    # 완만한 경사로 곡선(κ~0.037)까지 보호하면 옆이동이 0으로 눌려 중앙에 머문다.
    # 급격한 포켓 탈출(κ>0.1)만 보호하도록 임계값을 높인다.
    ap.add_argument("--k-thr", type=float, default=0.09, help="이 곡률 이상 구간만 옆이동 보호(급회전만)")
    ap.add_argument("--smooth-m", type=float, default=5.0, help="옆이동 전이 스무딩 거리[m]")
    ap.add_argument("--out-json", default="geometry/reference_path_keepright.json")
    ap.add_argument("--out-png", default="outputs/keepright_exit.png")
    args = ap.parse_args()

    env = BusHairpinEnv()
    wb, hw = env.model.L, env.half_width
    kappa_max = 1.0 / env.model.R
    d = json.load(open(SIM_DIR / "geometry" / "reference_path_wheelmodel.json", encoding="utf-8"))
    rw = d["rear_waypoints"]; gears = d["gears"]
    runs = split_runs(rw, gears)
    a4 = runs[-1][0]
    exit_poses = [tuple(p) for p in rw[a4:]]
    N = len(exit_poses)

    # arc-length, 원본 곡률 -> 직진도(곡선 보호)
    s = [0.0]
    for i in range(1, N):
        s.append(s[-1] + math.hypot(exit_poses[i][0] - exit_poses[i - 1][0],
                                    exit_poses[i][1] - exit_poses[i - 1][1]))
    Ltot = s[-1]
    kap = [0.0] * N
    for i in range(1, N - 1):
        dss = s[i + 1] - s[i - 1]
        if dss > 1e-6:
            kap[i] = abs(_wrap(exit_poses[i + 1][2] - exit_poses[i - 1][2])) / dss
    straight = [max(0.0, min(1.0, (args.k_thr - k) / args.k_thr)) for k in kap]
    straight[0] = 0.0  # 시작(2차후진 직후)은 회차와 연속 유지

    # offset 프로파일: 직진도*offset, 끝(도착)은 오른쪽차선까지 유지(꼬리 taper 없음)
    raw = [args.offset_right * straight[i] for i in range(N)]
    avg_ds = Ltot / max(1, N - 1)
    win = max(1, int(round(args.smooth_m / avg_ds)))
    off = []
    for i in range(N):
        lo, hi = max(0, i - win), min(N, i + win + 1)
        off.append(sum(raw[lo:hi]) / (hi - lo))
    off[0] = 0.0

    # 오른쪽으로 이동 (heading -90°)
    shifted = []
    for i, (x, y, th) in enumerate(exit_poses):
        rx, ry = math.sin(th), -math.cos(th)
        shifted.append([x + off[i] * rx, y + off[i] * ry])
    shifted[0] = [exit_poses[0][0], exit_poses[0][1]]
    # 끝점: 오른쪽 차선 도착 (캡중앙 뒷축 + offset*오른쪽), heading 원본 유지
    gx, gy, gth = exit_poses[-1]
    grx, gry = math.sin(gth), -math.cos(gth)
    shifted[-1] = [gx + args.offset_right * grx, gy + args.offset_right * gry]

    # heading 재계산 (옮긴 폴리라인 접선; 곡선 보호구간은 원본과 거의 동일)
    new_exit = []
    for i in range(N):
        if i == 0:
            th = exit_poses[0][2]
        elif i == N - 1:
            th = exit_poses[-1][2]
        else:
            th = math.atan2(shifted[i + 1][1] - shifted[i - 1][1],
                            shifted[i + 1][0] - shifted[i - 1][0])
        new_exit.append([round(shifted[i][0], 3), round(shifted[i][1], 3), round(th, 6)])

    # 곡률 검사
    max_kappa = 0.0
    for i in range(1, N):
        ds = math.hypot(new_exit[i][0] - new_exit[i - 1][0], new_exit[i][1] - new_exit[i - 1][1])
        if ds > 1e-6:
            max_kappa = max(max_kappa, abs(_wrap(new_exit[i][2] - new_exit[i - 1][2])) / ds)

    # 접합 + 바퀴 검증
    new_rw = [list(rw[i]) for i in range(a4)] + new_exit
    new_gears = gears[:a4] + ["FWD"] * N
    viol = sum(1 for (x, y, th) in new_rw if not env.boundary_polygon.contains(wheel_poly(x, y, th, wb, hw)))
    n_switch = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    total_len = sum(math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
                    for i in range(1, len(new_rw)))
    gfc = (new_rw[-1][0] + env.d_ref * math.cos(new_rw[-1][2]),
           new_rw[-1][1] + env.d_ref * math.sin(new_rw[-1][2]))
    print(f"진출 노드 {N}, 길이 {Ltot:.1f}m, offset {args.offset_right}m")
    print(f"최대 곡률 {max_kappa:.4f} (한계 {kappa_max:.4f}) -> {'OK' if max_kappa <= kappa_max * 1.05 else '초과!'}")
    print(f"접합 후: 노드 {len(new_rw)}, 방향전환 {n_switch}, 길이 {total_len:.1f}m, 바퀴위반 {viol}")
    print(f"도착 앞면중심 {gfc[0]:.2f},{gfc[1]:.2f} (캡중앙 {env.goal_x},{env.goal_y} 대비 오른쪽 {args.offset_right}m)")
    if viol > 0 or max_kappa > kappa_max * 1.05:
        print("경고: 위반/곡률초과 -> --offset-right 줄이거나 --smooth-m 늘릴 것 (저장은 함)")

    waypoints = [[round(x + env.d_ref * math.cos(th), 3), round(y + env.d_ref * math.sin(th), 3)]
                 for (x, y, th) in new_rw]
    out = {"waypoints": waypoints, "gears": new_gears,
           "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw],
           "collision_model": "wheel"}
    json.dump(out, open(SIM_DIR / args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장:", args.out_json)

    # 시각화
    fig, ax = plt.subplots(figsize=(13, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", lw=1.2)
    ox = [p[0] for p in exit_poses]; oy = [p[1] for p in exit_poses]
    ax.plot(ox, oy, "b--", lw=1.3, alpha=0.55, label="이전 진출(좌측통행)")
    for i in range(1, len(new_rw)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax.plot([new_rw[i - 1][0], new_rw[i][0]], [new_rw[i - 1][1], new_rw[i][1]], color=color, lw=1.9, zorder=5)
    ax.plot([], [], color="tab:blue", lw=2, label="전진(우측통행)")
    ax.plot([], [], color="tab:red", lw=2, label="후진")
    ax.plot(new_rw[0][0], new_rw[0][1], "g^", ms=12, zorder=6)
    ax.plot(env.goal_x, env.goal_y, "kx", ms=12, mew=2, label="캡 중앙", zorder=6)
    ax.plot(gfc[0], gfc[1], "m*", ms=15, label="도착(오른쪽 차선)", zorder=6)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title(f"우측통행 진출(오른쪽차선 {args.offset_right}m): 방향전환 {n_switch}회, "
                 f"길이 {total_len:.1f}m, 바퀴위반 {viol}, 최대곡률 {max_kappa:.3f}/{kappa_max:.3f}")
    fig.tight_layout()
    fig.savefig(SIM_DIR / args.out_png, dpi=150)
    print("저장:", args.out_png)


if __name__ == "__main__":
    main()
