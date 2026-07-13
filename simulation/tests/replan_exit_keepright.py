"""우측통행(keep-right): 회차(run0~run3, 2차 후진까지)는 그대로 두고, 마지막
전진 진출구간(run4)만 오른쪽에 붙게 다시 계획한다.

진출구간은 현재 좌측통행(왼쪽여유 3.7m << 오른쪽여유 6.7m)이다. 진출로
'중간 구간'의 왼쪽(진행방향 기준 좌측)에 keep-out 띠를 obstacle로 넣어
버스를 오른쪽으로 민다. 시작(포켓 탈출)·도착(캡 중앙) 부근은 keep-out을
비워 자유롭게 둔다. 목표점(경사로 끝단 중앙)은 고정.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_DIR))
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState
from planning.hybrid_astar import HybridAStarPlanner



def split_runs(rw, gears):
    n = len(rw); edge = gears[1:n]; runs = []; k = 0; m = len(edge)
    while k < m:
        g = edge[k]; j = k
        while j < m and edge[j] == g:
            j += 1
        runs.append((k, j, g)); k = j
    return runs


def _longest_line(geom):
    if geom.geom_type == "LineString":
        return geom
    return max(geom.geoms, key=lambda g: g.length)


def _extend(coords, ext=40.0):
    """폴리라인 양끝을 끝 접선 방향으로 ext[m] 연장(도로 밖까지 빼서 단측
    버퍼의 끝단 말림이 도로 안으로 들어오지 않게)."""
    c = list(coords)
    (x0, y0), (x1, y1) = c[0], c[1]
    a = math.atan2(y0 - y1, x0 - x1)
    (xn, yn), (xm, ym) = c[-1], c[-2]
    b = math.atan2(yn - ym, xn - xm)
    return [(x0 + ext * math.cos(a), y0 + ext * math.sin(a))] + c + [(xn + ext * math.cos(b), yn + ext * math.sin(b))]


def build_keepout(exit_xy, road_poly, offset_right, clear_radius):
    """진출 경로를 오른쪽으로 offset_right 옮긴 median 기준, 그 왼쪽 도로
    영역을 keep-out으로 만들되 시작/목표 주변 clear_radius 원은 비운다."""
    from shapely.geometry import Point
    line = LineString(exit_xy)
    i = len(exit_xy) // 2
    inxt = min(i + 1, len(exit_xy) - 1)
    ang = math.atan2(exit_xy[inxt][1] - exit_xy[i - 1][1], exit_xy[inxt][0] - exit_xy[i - 1][0])
    right_n = (math.cos(ang - math.pi / 2), math.sin(ang - math.pi / 2))  # 진행방향 우측
    best = None
    for dist in (offset_right, -offset_right):
        try:
            oc = _longest_line(line.offset_curve(dist))
        except Exception:
            continue
        mp = oc.interpolate(oc.length / 2).coords[0]
        dv = (mp[0] - exit_xy[i][0], mp[1] - exit_xy[i][1])
        dot = dv[0] * right_n[0] + dv[1] * right_n[1]
        if best is None or dot > best[0]:
            best = (dot, oc)
    median = best[1]

    # median 왼쪽(진행방향 좌측 = 현재 좌측통행 경로 쪽)에 '국소' 좁은 띠만.
    # 이전엔 median을 연장 후 200m 단측버퍼를 씌워, 반평면이 진입램프·먼
    # 영역까지 다 잡아먹었다. 좁은 띠(strip_w) + 진출통로 근처로 클립해 국한한다.
    midpt = Point(exit_xy[i][0], exit_xy[i][1])
    strip = None
    for d in (6.0, -6.0):
        cand = median.buffer(d, single_sided=True)
        if cand.contains(midpt):
            strip = cand
            break
    if strip is None:
        strip = median.buffer(6.0, single_sided=True)

    corridor = LineString(exit_xy).buffer(10.0)  # 진출 통로 근처로만 국한
    keepout = strip.intersection(road_poly).intersection(corridor)

    s, g = exit_xy[0], exit_xy[-1]
    keepout = (keepout
               .difference(Point(*s).buffer(clear_radius))
               .difference(Point(*g).buffer(clear_radius)))
    return keepout, median


def add_geom(ax, geom, **kw):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            cs = list(ring.coords)
            verts += cs
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(cs) - 2) + [MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-right", type=float, default=2.4, help="진출 median 우측 이동량[m]")
    ap.add_argument("--clear-radius", type=float, default=7.0, help="시작/목표 주변 keep-out 비우는 반경[m]")
    # 진출 통로는 넓어 거친 격자로도 충분 -> 빠르게. (전체 회차는 fine이 필요했지만 여기 진출만은 아님)
    ap.add_argument("--arc-len", type=float, default=0.5)
    ap.add_argument("--xy-res", type=float, default=0.35)
    ap.add_argument("--theta-res", type=float, default=4.0)
    ap.add_argument("--steers", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=120000)
    ap.add_argument("--dry-run", action="store_true", help="계획 없이 keep-out 모양만 그려 진단")
    ap.add_argument("--out-json", default="geometry/reference_path_keepright.json")
    ap.add_argument("--out-png", default="outputs/keepright_exit.png")
    args = ap.parse_args()

    env = BusHairpinEnv()
    wheel_fp = (env.model.L, 0.0, env.half_width)
    d = json.load(open(SIM_DIR / "geometry" / "reference_path_wheelmodel.json", encoding="utf-8"))
    rw = d["rear_waypoints"]; gears = d["gears"]
    runs = split_runs(rw, gears)
    a4 = runs[-1][0]  # 진출 leg 시작 노드 (2차 후진 직후)

    exit_xy = [(x, y) for (x, y, th) in rw[a4:]]
    keepout, median = build_keepout(exit_xy, env.boundary_polygon,
                                    args.offset_right, args.clear_radius)
    # keep-out은 경계(16000+정점)에서 잘려 정점이 많다 -> 매 노드 obstacle
    # 충돌검사가 느려지므로 단순화(형상 거의 불변, 속도만 크게 개선).
    keepout = keepout.simplify(0.15, preserve_topology=True)
    print(f"진출 leg 시작 노드 {a4}, keep-out 면적 {keepout.area:.1f} m^2")

    exit_start = tuple(rw[a4])
    goal_rear = (env.goal_x - env.d_ref * math.cos(env.goal_heading),
                 env.goal_y - env.d_ref * math.sin(env.goal_heading),
                 env.goal_heading)

    if args.dry_run:
        import matplotlib
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 9))
        px, py = env.boundary_polygon.exterior.xy
        ax.plot(px, py, "k-", lw=1.2)
        add_geom(ax, keepout, facecolor="#BBBBBB", edgecolor="#666", alpha=0.6, zorder=2)
        ex = [p[0] for p in exit_xy]; ey = [p[1] for p in exit_xy]
        ax.plot(ex, ey, "b--", lw=1.5, label="현재 진출경로(좌측통행)", zorder=4)
        mx, my = median.xy
        ax.plot(mx, my, "m:", lw=1.5, label="median(우측 이동)", zorder=4)
        ax.plot(exit_start[0], exit_start[1], "g^", ms=13, label="진출 시작(2차후진 직후)", zorder=6)
        ax.plot(goal_rear[0], goal_rear[1], "kx", ms=13, mew=2, label="목표(뒷축)", zorder=6)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_aspect("equal")
        ax.set_title(f"[진단] keep-out 면적 {keepout.area:.1f} m² (offset_right={args.offset_right}, clear_radius={args.clear_radius})")
        fig.tight_layout()
        fig.savefig(SIM_DIR / "outputs" / "keepright_dryrun.png", dpi=150)
        print("저장: outputs/keepright_dryrun.png (진단용)")
        return

    # keep-out이 시작/목표를 덮지 않는지 확인
    from shapely.geometry import Point
    for nm, p in [("start", exit_start), ("goal", goal_rear)]:
        if keepout.contains(Point(p[0], p[1])):
            print(f"경고: keep-out이 {nm} 지점을 덮음 -> trim 늘려야 함")

    obstacles = list(env.obstacle_polygons) + [keepout]
    steer = tuple(round(-1.0 + 2.0 * i / (args.steers - 1), 3) for i in range(args.steers))
    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon, footprint=wheel_fp,
        delta_max=env.model.delta_max, arc_len=args.arc_len, steer_fracs=steer,
        xy_res=args.xy_res, theta_res_deg=args.theta_res, reverse_penalty=2.0, gear_switch_penalty=5.0,
        tol_pos=0.6, tol_heading_deg=12.0, obstacle_polygons=obstacles, debug=True,
    )

    import time
    t0 = time.time()
    exit_path = planner.plan(exit_start, goal_rear, L=env.model.L, max_iter=args.max_iter, bidirectional=True)
    print("진출 재계획 elapsed", round(time.time() - t0, 1), "s")
    if exit_path is None:
        print("진출 재계획 실패 (offset/ trim 조정 필요)")
        return

    # 접합: prefix(run0~3) + 새 진출 leg (중복 시작점 제외)
    new_rw = [list(rw[i]) for i in range(a4 + 1)]
    new_gears = gears[:a4 + 1]
    for (x, y, th, g) in exit_path[1:]:
        new_rw.append([round(x, 3), round(y, 3), round(th, 6)])
        new_gears.append("FWD" if g == "START" else g)

    # 전수 검증(바퀴 footprint)
    def wheel_poly(x, y, th):
        c, s = math.cos(th), math.sin(th)
        f, r, hw = wheel_fp
        loc = [(f, hw), (f, -hw), (-r, -hw), (-r, hw)]
        return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in loc])
    viol = sum(1 for (x, y, th) in new_rw if not env.boundary_polygon.contains(wheel_poly(x, y, th)))
    ko_hit = sum(1 for (x, y, th) in new_rw if wheel_poly(x, y, th).intersects(keepout))
    n_switch = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    total_len = sum(math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
                    for i in range(1, len(new_rw)))
    print(f"접합 후: 노드 {len(new_rw)}, 방향전환 {n_switch}, 길이 {total_len:.1f}m, "
          f"바퀴위반 {viol}, keepout침범 {ko_hit}")

    waypoints = []
    for (x, y, th) in new_rw:
        waypoints.append([round(x + env.d_ref * math.cos(th), 3), round(y + env.d_ref * math.sin(th), 3)])
    out = {"waypoints": waypoints, "gears": new_gears,
           "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw],
           "collision_model": "wheel"}
    json.dump(out, open(SIM_DIR / args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장:", args.out_json)

    # 시각화
    import matplotlib
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", lw=1.2)
    add_geom(ax, keepout, facecolor="#BBBBBB", edgecolor="none", alpha=0.55, zorder=2)
    for i in range(1, len(new_rw)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax.plot([new_rw[i - 1][0], new_rw[i][0]], [new_rw[i - 1][1], new_rw[i][1]], color=color, lw=1.8, zorder=5)
    ax.plot(new_rw[0][0], new_rw[0][1], "g^", ms=12, zorder=6)
    ax.plot(env.goal_x, env.goal_y, "kx", ms=13, mew=2, zorder=6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor="#BBBBBB", alpha=0.55, label="keep-out (좌측 금지 띠)"),
        Line2D([0], [0], color="tab:blue", lw=2, label="전진"),
        Line2D([0], [0], color="tab:red", lw=2, label="후진"),
    ], loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title(f"우측통행 진출 재계획: 방향전환 {n_switch}회, 길이 {total_len:.1f}m, 바퀴위반 {viol}")
    fig.tight_layout()
    fig.savefig(SIM_DIR / args.out_png, dpi=150)
    print("저장:", args.out_png)


if __name__ == "__main__":
    main()
