"""회차 경로를 따라 버스 차체가 쓸고 지나간 스윕 영역으로 차단/주의 구역을 그린다.

- 차단구역(주황): 경로 전 구간에서 버스 차체(범퍼 포함 전체 외곽)의 합집합.
  = 버스가 실제로 점유했던 공간. Vehicle Tracking의 swept path와 같은 개념.
- 주의구역(노랑): 차단구역을 바깥으로 buffer 0.5m 확장한 영역.
- 경로선: 전진 파랑 / 후진 빨강 (기존 결과 PNG와 동일 규약).
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon
from shapely.ops import unary_union

# 한글 라벨이 깨지지 않도록 Windows 기본 한글 폰트 지정.
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from envs.bus_hairpin_env import BusHairpinEnv

SIM_DIR = Path(__file__).resolve().parent.parent

CAUTION_BUFFER_M = 0.5


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_polygon(x, y, theta, front, rear, half_w):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front, half_w), (front, -half_w), (-rear, -half_w), (-rear, half_w)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])


def densify(rear_wps, substeps=4):
    """인접 뒷축 자세 사이를 substeps로 선형(각도는 wrap)보간해 스윕 틈을 없앤다."""
    out = []
    for i in range(len(rear_wps) - 1):
        x0, y0, t0 = rear_wps[i]
        x1, y1, t1 = rear_wps[i + 1]
        dt = _wrap(t1 - t0)
        for k in range(substeps):
            f = k / substeps
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, t0 + dt * f))
    out.append(tuple(rear_wps[-1]))
    return out


def _polys(geom):
    """임의 geometry에서 Polygon만 추출(교집합이 GeometryCollection이어도 안전)."""
    if geom.is_empty:
        return []
    t = geom.geom_type
    if t == "Polygon":
        return [geom]
    if t in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polys(g))
        return out
    return []


def add_geom(ax, geom, **kw):
    """shapely (Multi)Polygon을 구멍까지 반영해 채워 그린다."""
    for poly in _polys(geom):
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            cs = list(ring.coords)
            verts.extend(cs)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(cs) - 2) + [MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(verts, codes), **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-json", default="geometry/reference_path_wheelmodel.json")
    ap.add_argument("--out", default="outputs/zones_wheelmodel.png")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--caution-buffer", type=float, default=2.0,
                    help="주의구역을 차단구역에서 바깥으로 확장할 폭[m]")
    ap.add_argument("--n-boxes", type=int, default=60,
                    help="경로 따라 그릴 차체 외곽(네모) 개수 (0=안 그림)")
    args = ap.parse_args()
    cb = args.caution_buffer

    env = BusHairpinEnv()
    front, rear, half_w = env.front_oh, env.rear_oh, env.half_width  # 차체(범퍼 포함)

    with open(SIM_DIR / args.path_json, encoding="utf-8") as f:
        data = json.load(f)
    rear_wps = data["rear_waypoints"]
    gears = data["gears"]

    # 차단구역: 차체 스윕 합집합. 주의구역: 바깥 0.5m 링(차단 제외).
    poses = densify(rear_wps, args.substeps)
    bodies = [body_polygon(x, y, t, front, rear, half_w) for (x, y, t) in poses]
    blocked = unary_union(bodies)
    ring = blocked.buffer(cb).difference(blocked)

    # 노면(경계 폴리곤) 밖으로 나가는 부분은 색칠하지 않도록 잘라낸다.
    blocked = blocked.intersection(env.boundary_polygon)
    ring = ring.intersection(env.boundary_polygon)

    print(f"경로: {args.path_json}  (노드 {len(rear_wps)}, 스윕 자세 {len(poses)})")
    print(f"차단구역 면적(노면 내): {blocked.area:.1f} m^2")
    print(f"주의구역 링 면적(+{cb}m, 노면 내): {ring.area:.1f} m^2")

    fig, ax = plt.subplots(figsize=(13, 9))

    # 노면 경계
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1.2, zorder=1)

    # 주의 링(노랑)과 차단(주황)은 겹치지 않게 분리해 각각 투명도 50%로.
    add_geom(ax, ring, facecolor="#FFD24D", edgecolor="#E0A800", linewidth=0.8, alpha=0.5, zorder=2)
    add_geom(ax, blocked, facecolor="#FF8A33", edgecolor="#D9660D", linewidth=0.8, alpha=0.5, zorder=3)

    # 차체 외곽(네모)을 경로 따라 일정 간격으로 그려 '차량 움직임'을 보이게 한다
    # (이 네모들이 쓸고 간 합집합이 곧 주황 차단구역).
    if args.n_boxes > 0:
        step_box = max(1, len(poses) // args.n_boxes)
        for i in range(0, len(poses), step_box):
            x, y, t = poses[i]
            bx, by = body_polygon(x, y, t, front, rear, half_w).exterior.xy
            ax.plot(bx, by, color="0.35", linewidth=0.5, zorder=4)

    # 경로선: 전진 파랑 / 후진 빨강 (뒷축 기준)
    for i in range(1, len(rear_wps)):
        color = "tab:red" if gears[i] == "REV" else "tab:blue"
        ax.plot([rear_wps[i - 1][0], rear_wps[i][0]], [rear_wps[i - 1][1], rear_wps[i][1]],
                color=color, linewidth=1.8, zorder=5)

    # 범례 프록시
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor="#FF8A33", edgecolor="#D9660D", alpha=0.5, label="차단구역 (버스 점유 스윕)"),
        Patch(facecolor="#FFD24D", edgecolor="#E0A800", alpha=0.5, label=f"주의구역 (+{cb:g}m)"),
        Line2D([0], [0], color="0.35", lw=0.8, label="차체 외곽(네모)"),
        Line2D([0], [0], color="tab:blue", lw=2.2, label="전진 (FWD)"),
        Line2D([0], [0], color="tab:red", lw=2.2, label="후진 (REV)"),
    ], loc="upper right", fontsize=10)

    # 시작/목표
    ax.plot(rear_wps[0][0], rear_wps[0][1], "g^", markersize=12, zorder=6)
    ax.plot(env.goal_x, env.goal_y, "kx", markersize=13, mew=2, zorder=6)

    ax.set_aspect("equal")
    n_switch = sum(1 for i in range(1, len(gears)) if gears[i] != gears[i - 1])
    ax.set_title(f"차단/주의 구역 (노면 내, 경로 방향전환 {n_switch}회, "
                 f"차단 {blocked.area:.0f} m² / 주의 링 {ring.area:.0f} m²)")
    fig.tight_layout()
    out_path = SIM_DIR / args.out
    fig.savefig(out_path, dpi=150)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
