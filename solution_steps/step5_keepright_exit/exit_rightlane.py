"""우측통행 진출 (lane-split) — 윗 경사로를 중앙선으로 2차선으로 가르고, 마지막
전진 진출구간(run4)을 '오른쪽 차선'에만 두도록 경로를 옮긴다.

각 진출 지점에서 진행방향 좌/우로 벽까지 ray를 쏴 코리도어 폭을 재고, 그
지점의 오른쪽 차선 중앙(=우측벽에서 폭의 1/4 지점)으로 경로를 옮긴다. 도로
폭이 변해도 항상 오른쪽 차선을 따른다. 급격한 포켓 탈출 구간(κ 큰 곳)은
보호(안 옮김)하고 전이는 스무딩. 곡률·바퀴위반 재검증.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

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


def ray_hit(bnd, x, y, ang, maxd=40.0):
    fx, fy = x + maxd * math.cos(ang), y + maxd * math.sin(ang)
    inter = bnd.intersection(LineString([(x, y), (fx, fy)]))
    if inter.is_empty:
        return None, None
    pts = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
    pts = [p for p in pts if p.geom_type == "Point"]
    if not pts:
        return None, None
    p = min(pts, key=lambda q: (q.x - x) ** 2 + (q.y - y) ** 2)
    return (p.x, p.y), math.hypot(p.x - x, p.y - y)


def smooth_spline(xs, ys, s, k=3):
    """scipy 없이: 가벼운 반복 라플라시안 스무딩(끝점 고정) + 접선 heading.
    s를 스무딩 세기(반복 횟수)로 매핑. 과하면 곡선이 평탄해지므로 가볍게 둔다."""
    xs = list(xs); ys = list(ys)
    n = len(xs)
    if n < 3:
        th = [math.atan2(ys[min(n - 1, i + 1)] - ys[max(0, i - 1)],
                         xs[min(n - 1, i + 1)] - xs[max(0, i - 1)]) for i in range(n)]
        return xs, ys, th
    iters = max(1, min(40, int(round(s * 3))))
    alpha = 0.35
    for _ in range(iters):
        nx, ny = xs[:], ys[:]
        for i in range(1, n - 1):
            nx[i] = xs[i] + alpha * (0.5 * (xs[i - 1] + xs[i + 1]) - xs[i])
            ny[i] = ys[i] + alpha * (0.5 * (ys[i - 1] + ys[i + 1]) - ys[i])
        xs, ys = nx, ny
    th = []
    for i in range(n):
        j1, j2 = max(0, i - 2), min(n - 1, i + 2)
        th.append(math.atan2(ys[j2] - ys[j1], xs[j2] - xs[j1]))
    return xs, ys, th


def march_corridor(bnd, x, y, thdir, step=0.5, max_steps=90, w_max=11.0):
    """(x,y)에서 thdir 방향으로 코리도어를 따라 전진하며 좌/우 벽 중점(중앙선)을
    추적한다. 각 스텝 (center, left_wall, right_wall). 폭이 w_max를 넘거나 벽을
    못 찾으면 코리도어를 벗어난 것으로 보고 멈춘다."""
    out = []
    for _ in range(max_steps):
        rn = (math.sin(thdir), -math.cos(thdir))
        ln = (-rn[0], -rn[1])
        lp, dL = ray_hit(bnd, x, y, math.atan2(ln[1], ln[0]))
        rp, dR = ray_hit(bnd, x, y, math.atan2(rn[1], rn[0]))
        if lp is None or rp is None or (dL + dR) > w_max:
            break
        cx, cy = (lp[0] + rp[0]) / 2, (lp[1] + rp[1]) / 2
        out.append(((cx, cy), lp, rp))
        x, y = cx + step * math.cos(thdir), cy + step * math.sin(thdir)
        if len(out) >= 2:
            thdir = math.atan2(out[-1][0][1] - out[-2][0][1], out[-1][0][0] - out[-2][0][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-frac", type=float, default=0.75,
                    help="좌측벽으로부터 목표 가로위치 비율 (0.75=오른쪽차선 중앙, 0.5=도로중앙)")
    ap.add_argument("--k-thr", type=float, default=0.08, help="이 곡률 미만이면 경사로로 보고 100% 이동(급회전만 보호)")
    ap.add_argument("--right-from", type=float, default=0.34,
                    help="램프의 이 비율 지점까지 전이 완료(그 전엔 좌측차로 허용), 이후 오른쪽차선 중앙 유지")
    ap.add_argument("--smooth-s", type=float, default=2.0,
                    help="램프 구간 스무딩 스플라인 강도(0=미적용, 클수록 매끄럽지만 차선에서 더 벗어남)")
    ap.add_argument("--out-json", default="geometry/reference_path_keepright.json")
    ap.add_argument("--out-png", default="outputs/keepright_exit.png")
    args = ap.parse_args()

    env = BusHairpinEnv()
    wb, hw = env.model.L, env.half_width
    kappa_max = 1.0 / env.model.R
    bnd = LineString(env.boundary_polygon.exterior.coords).simplify(0.05)
    d = json.load(open(SIM_DIR / "geometry" / "reference_path_wheelmodel.json", encoding="utf-8"))
    rw = d["rear_waypoints"]; gears = d["gears"]
    runs = split_runs(rw, gears)
    a4 = runs[-1][0]
    ep = [tuple(p) for p in rw[a4:]]
    N = len(ep)

    # arc-length, 곡률, 직진도
    s = [0.0]
    for i in range(1, N):
        s.append(s[-1] + math.hypot(ep[i][0] - ep[i - 1][0], ep[i][1] - ep[i - 1][1]))
    Ltot = s[-1]
    kap = [0.0] * N
    for i in range(1, N - 1):
        dss = s[i + 1] - s[i - 1]
        if dss > 1e-6:
            kap[i] = abs(_wrap(ep[i + 1][2] - ep[i - 1][2])) / dss
    # 급회전 포켓(맨 앞의 초기 급회전)이 끝나는 arc-length. '마지막' κ>=k_thr을
    # 쓰면 목표 근처의 곡선 하나까지 잡아 게이트가 전 구간 꺼진다 -> 초기 급회전
    # 이후 3m 넘게 완만하면 거기서 포켓이 끝난 것으로 확정.
    pocket_end_s = 0.0
    for i in range(N):
        if kap[i] >= args.k_thr:
            pocket_end_s = s[i]
        elif s[i] - pocket_end_s > 3.0:
            break

    # 각 지점: 좌/우 벽까지 거리 + 오른쪽 차선 중앙까지 목표 이동량
    left_wall, right_wall, center_pts = [], [], []
    target = [0.0] * N
    dL_arr = [None] * N
    W_arr = [None] * N
    for i, (x, y, th) in enumerate(ep):
        rnx, rny = math.sin(th), -math.cos(th)          # 우측(진행 -90°)
        lnx, lny = -rnx, -rny                            # 좌측
        lp, dL = ray_hit(bnd, x, y, math.atan2(lny, lnx))
        rp, dR = ray_hit(bnd, x, y, math.atan2(rny, rnx))
        left_wall.append(lp); right_wall.append(rp)
        if dL is not None and dR is not None:
            W = dL + dR
            dL_arr[i] = dL; W_arr[i] = W
            center_pts.append((x + (dR - dL) / 2 * rnx, y + (dR - dL) / 2 * rny))
            target[i] = args.lane_frac * W - dL       # 오른쪽차선 중앙(좌측벽 기준 lane_frac*W)까지
        else:
            center_pts.append((x, y))

    # target은 ray 잡음만 살짝 제거(작은 2m 창) — 평탄역은 유지(희석 금지).
    avg_ds = Ltot / max(1, N - 1)
    tw = max(1, int(round(2.0 / avg_ds)))
    target_s = [sum(target[max(0, i - tw):min(N, i + tw + 1)]) / len(target[max(0, i - tw):min(N, i + tw + 1)])
                for i in range(N)]

    # 오른쪽 차선 중앙점(램프 노드, 폭<9m): ep 위치 + 목표 가로이동.
    rlc = []  # (s, x, y, heading)
    for i in range(N):
        if W_arr[i] and W_arr[i] < 9.0:
            rnx, rny = math.sin(ep[i][2]), -math.cos(ep[i][2])
            rlc.append((s[i], ep[i][0] + target_s[i] * rnx, ep[i][1] + target_s[i] * rny, ep[i][2]))
    if not rlc:
        raise SystemExit("램프 구간(폭<9m)을 못 찾음")
    ramp_s0 = rlc[0][0]
    s13 = ramp_s0 + args.right_from * (Ltot - ramp_s0)
    p1s, p1x, p1y, p1h = min(rlc, key=lambda r: abs(r[0] - s13))
    print(f"포켓 끝 arc={pocket_end_s:.1f}m, 램프 시작 s={ramp_s0:.1f}m, "
          f"완만한 곡선 종료(1/3) s={p1s:.1f}m, 목표 가로비율={args.lane_frac}")

    # 시작점(heading 고정) -> 램프 1/3 오른쪽차선점(heading=램프 방향)을 하나의
    # Hermite 곡선으로. 차선변경이 아니라 처음부터 완만히 오른쪽 차선으로 흘러감.
    P0 = (ep[0][0], ep[0][1]); h0 = ep[0][2]
    P1 = (p1x, p1y); h1 = p1h
    chord = math.hypot(P1[0] - P0[0], P1[1] - P0[1])
    T0 = (chord * math.cos(h0), chord * math.sin(h0))
    T1 = (chord * math.cos(h1), chord * math.sin(h1))
    nseg = max(2, int(round(1.4 * chord / 0.3)))
    curve = []
    for kk in range(nseg + 1):
        t = kk / nseg
        a00 = 2 * t**3 - 3 * t**2 + 1; a10 = t**3 - 2 * t**2 + t
        a01 = -2 * t**3 + 3 * t**2; a11 = t**3 - t**2
        curve.append((a00 * P0[0] + a10 * T0[0] + a01 * P1[0] + a11 * T1[0],
                      a00 * P0[1] + a10 * T0[1] + a01 * P1[1] + a11 * T1[1]))
    # 1/3 이후: 오른쪽 차선 중앙점
    tail = [(rx, ry) for (rs_, rx, ry, rh) in rlc if rs_ > p1s]
    pts = curve + tail

    # 전체를 스무딩 스플라인으로 한 번 더 다듬어 곡선-직선 이음매까지 매끄럽게.
    if args.smooth_s > 0 and len(pts) >= 4:
        sx, sy, sth = smooth_spline([p[0] for p in pts], [p[1] for p in pts], args.smooth_s)
        sx[0], sy[0] = pts[0]; sx[-1], sy[-1] = pts[-1]
        pts2, ths = list(zip(sx, sy)), sth
    else:
        pts2, ths = pts, None

    M = len(pts2)
    new_exit = []
    for i in range(M):
        if i == 0:
            th = h0
        elif i == M - 1:
            th = ep[-1][2]
        elif ths is not None:
            th = ths[i]
        else:
            th = math.atan2(pts2[i + 1][1] - pts2[i - 1][1], pts2[i + 1][0] - pts2[i - 1][0])
        new_exit.append([round(pts2[i][0], 3), round(pts2[i][1], 3), round(th, 6)])
    Nx = len(new_exit)  # 새 진출 노드 수(원본 N=len(ep)와 다를 수 있음; 시각화는 원본 N 사용)

    max_kappa = 0.0
    for i in range(1, Nx):
        ds = math.hypot(new_exit[i][0] - new_exit[i - 1][0], new_exit[i][1] - new_exit[i - 1][1])
        if ds > 1e-6:
            max_kappa = max(max_kappa, abs(_wrap(new_exit[i][2] - new_exit[i - 1][2])) / ds)

    new_rw = [list(rw[i]) for i in range(a4)] + new_exit
    new_gears = gears[:a4] + ["FWD"] * Nx
    viol = sum(1 for (x, y, th) in new_rw if not env.boundary_polygon.contains(wheel_poly(x, y, th, wb, hw)))
    n_switch = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    total_len = sum(math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
                    for i in range(1, len(new_rw)))
    gfc = (new_rw[-1][0] + env.d_ref * math.cos(new_rw[-1][2]),
           new_rw[-1][1] + env.d_ref * math.sin(new_rw[-1][2]))
    print(f"진출 노드 {Nx}, 길이 {Ltot:.1f}m, lane_frac {args.lane_frac} (0.75=오른쪽차선중앙)")
    print(f"최대 곡률 {max_kappa:.4f} (한계 {kappa_max:.4f}) -> {'OK' if max_kappa <= kappa_max * 1.05 else '초과!'}")
    print(f"접합 후: 노드 {len(new_rw)}, 방향전환 {n_switch}, 길이 {total_len:.1f}m, 바퀴위반 {viol}")
    print(f"도착 앞면중심 {gfc[0]:.2f},{gfc[1]:.2f}")
    if viol > 0 or max_kappa > kappa_max * 1.05:
        print("경고: 위반/곡률초과 -> --chord 곡선을 늘리려면 --right-from 0.45, 또는 --lane-frac 0.72 (저장은 함)")

    waypoints = [[round(x + env.d_ref * math.cos(th), 3), round(y + env.d_ref * math.sin(th), 3)]
                 for (x, y, th) in new_rw]
    out = {"waypoints": waypoints, "gears": new_gears,
           "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw],
           "collision_model": "wheel"}
    json.dump(out, open(SIM_DIR / args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장:", args.out_json)

    # 시각화: 벽/차선분리선(중앙선)/좌측차선 음영 + 경로
    fig, ax = plt.subplots(figsize=(13, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", lw=1.2)
    # 윗 경사로 중앙선: 램프 구간(폭<9m)은 검증된 진출 노드 중점으로(junction
    # 곡선부 포함), 목표~캡 구간은 march로 연장(뒷축노드가 없는 구간). junction
    # 쪽에서 march를 시작하면 한쪽이 회차공간으로 열려 바로 멈추므로, 연장은
    # 반드시 '깨끗한' 목표 지점에서 캡 방향으로만 한다.
    ramp_idx = [i for i in range(N) if W_arr[i] and W_arr[i] < 9.0 and left_wall[i]]
    cl = [center_pts[i] for i in ramp_idx]
    lw_pts = [left_wall[i] for i in ramp_idx]
    ext = march_corridor(bnd, new_rw[-1][0], new_rw[-1][1], new_rw[-1][2])
    cl += [m[0] for m in ext]
    lw_pts += [m[1] for m in ext]
    if len(cl) > 1:
        ax.plot([p[0] for p in cl], [p[1] for p in cl], "g-.", lw=1.6, label="차선 분리선(중앙)")
        lane_poly = lw_pts + cl[::-1]
        ax.fill([p[0] for p in lane_poly], [p[1] for p in lane_poly],
                color="0.7", alpha=0.4, label="좌측 차선(주행 금지)")
    for i in range(1, len(new_rw)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax.plot([new_rw[i - 1][0], new_rw[i][0]], [new_rw[i - 1][1], new_rw[i][1]], color=color, lw=2.0, zorder=5)
    ax.plot([], [], color="tab:blue", lw=2.2, label="전진(오른쪽 차선)")
    ax.plot([], [], color="tab:red", lw=2.2, label="후진")
    ax.plot(new_rw[0][0], new_rw[0][1], "g^", ms=12, zorder=6)
    ax.plot(gfc[0], gfc[1], "m*", ms=15, label="도착(오른쪽 차선)", zorder=6)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(f"우측통행 진출(오른쪽 차선): 방향전환 {n_switch}회, 길이 {total_len:.1f}m, "
                 f"바퀴위반 {viol}, 최대곡률 {max_kappa:.3f}/{kappa_max:.3f}")
    fig.tight_layout()
    fig.savefig(SIM_DIR / args.out_png, dpi=150)
    print("저장:", args.out_png)


if __name__ == "__main__":
    main()
