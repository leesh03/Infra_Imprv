"""회차 경로를 Autodesk Vehicle Tracking용 DXF로 추출한다.

좌표계: 파이프라인(dxf_parser)이 원본 DXF(mm)를 ÷1000해 m로만 바꿨고(회전/
평행이동 없음), 따라서 여기서 ×1000 하면 원본 road_model.dxf(mm) 위에 정확히
겹친다. 기본 단위는 mm(원본과 동일). --units m 로 m 출력도 가능.

Vehicle Tracking은 폴리라인을 '전진'으로 따라가므로, 방향전환(전/후진 전환)
지점에서 경로를 끊어 **각 구간을 방향별 레이어의 개별 폴리라인**으로 넣는다
(사용자가 VT에서 해당 구간을 forward/reverse로 지정). 기준점을 무엇으로
쓰든 되도록 뒷축 중심 경로와 앞축 중심 경로를 둘 다 넣는다.
"""
import argparse
import json
import math
from pathlib import Path

import ezdxf

from envs.bus_hairpin_env import BusHairpinEnv

SIM_DIR = Path(__file__).resolve().parent.parent

LAYERS = {
    "ROAD_BOUNDARY":      {"color": 8},   # 회색
    "PATH_REAR_FWD":      {"color": 5},   # 파랑 (뒷축, 전진)
    "PATH_REAR_REV":      {"color": 1},   # 빨강 (뒷축, 후진)
    "PATH_FRONTAXLE_FWD": {"color": 4},   # 시안 (앞축, 전진)
    "PATH_FRONTAXLE_REV": {"color": 6},   # 마젠타 (앞축, 후진)
    "GEAR_CHANGE_PTS":    {"color": 2},   # 노랑 (방향전환 지점)
    "START_GOAL":         {"color": 3},   # 초록 (시작/목표 자세)
}


def split_runs(rear_wps, gears):
    """gears[i]는 노드 i로 들어오는 에지(i-1->i)의 기어(gears[0]은 시작 더미).
    같은 기어가 이어지는 최대 구간(run)들로 노드를 나눈다. 각 run은 경계
    노드를 이웃 run과 공유한다(끊긴 폴리라인이 cusp에서 만나도록)."""
    n = len(rear_wps)
    if n < 2:
        return []
    edge_gears = gears[1:n]
    runs = []
    k = 0
    m = len(edge_gears)
    while k < m:
        g = edge_gears[k]
        j = k
        while j < m and edge_gears[j] == g:
            j += 1
        runs.append((k, j, g))  # 노드 k..j (포함), 기어 g
        k = j
    return runs


def rect_pts(x, y, theta, front, rear, half_w, scale):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front, half_w), (front, -half_w), (-rear, -half_w), (-rear, half_w)]
    return [((x + lx * c - ly * s) * scale, (y + lx * s + ly * c) * scale) for lx, ly in local]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-json", default="geometry/reference_path_wheelmodel.json")
    ap.add_argument("--out", default="outputs/reference_path_wheelmodel.dxf")
    ap.add_argument("--units", choices=["mm", "m"], default="mm",
                    help="mm=원본 road_model.dxf와 동일 좌표(권장, 겹침), m=미터")
    args = ap.parse_args()

    scale = 1000.0 if args.units == "mm" else 1.0
    env = BusHairpinEnv()
    L = env.model.L
    d_ref = env.d_ref
    rear_oh = env.rear_oh
    half_w = env.half_width

    with open(SIM_DIR / args.path_json, encoding="utf-8") as f:
        data = json.load(f)
    rear_wps = data["rear_waypoints"]        # [x, y, theta] (뒷축, m)
    gears = data["gears"]

    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM if args.units == "mm" else ezdxf.units.M
    for name, attrs in LAYERS.items():
        doc.layers.add(name, color=attrs["color"])
    msp = doc.modelspace()

    # --- 도로 경계(문맥) ---
    bpts = [(px * scale, py * scale) for px, py in env.boundary_polygon.exterior.coords]
    msp.add_lwpolyline(bpts, close=True, dxfattribs={"layer": "ROAD_BOUNDARY"})

    # --- 뒷축/앞축 경로: 방향별 구간 폴리라인 ---
    def front_axle(x, y, th):
        return (x + L * math.cos(th), y + L * math.sin(th))

    runs = split_runs(rear_wps, gears)
    for (k, j, g) in runs:
        seg = rear_wps[k:j + 1]
        rear_line = [(x * scale, y * scale) for (x, y, th) in seg]
        fa_line = [(fx * scale, fy * scale) for (x, y, th) in seg for (fx, fy) in [front_axle(x, y, th)]]
        rear_layer = "PATH_REAR_REV" if g == "REV" else "PATH_REAR_FWD"
        fa_layer = "PATH_FRONTAXLE_REV" if g == "REV" else "PATH_FRONTAXLE_FWD"
        msp.add_lwpolyline(rear_line, dxfattribs={"layer": rear_layer})
        msp.add_lwpolyline(fa_line, dxfattribs={"layer": fa_layer})

    # --- 방향전환 지점 마커 ---
    marker_r = (env.model.L * 0.15) * scale
    for i in range(1, len(runs)):
        k = runs[i][0]
        x, y, th = rear_wps[k]
        msp.add_circle((x * scale, y * scale), marker_r, dxfattribs={"layer": "GEAR_CHANGE_PTS"})

    # --- 시작/목표 자세: 차체 외곽(범퍼 기준) 실루엣 + 라벨 ---
    for label, (x, y, th) in [("START", rear_wps[0]), ("GOAL", rear_wps[-1])]:
        body = rect_pts(x, y, th, d_ref, rear_oh, half_w, scale)
        msp.add_lwpolyline(body, close=True, dxfattribs={"layer": "START_GOAL"})
        # 진행방향 표시선(뒷축 -> 앞면)
        fx, fy = x + d_ref * math.cos(th), y + d_ref * math.sin(th)
        msp.add_line((x * scale, y * scale), (fx * scale, fy * scale), dxfattribs={"layer": "START_GOAL"})
        txt = msp.add_text(label, height=marker_r * 1.5, dxfattribs={"layer": "START_GOAL"})
        txt.set_placement((x * scale, y * scale))

    out_path = SIM_DIR / args.out
    doc.saveas(out_path)

    n_switch = len(runs) - 1
    total_len = sum(
        math.hypot(rear_wps[i][0] - rear_wps[i - 1][0], rear_wps[i][1] - rear_wps[i - 1][1])
        for i in range(1, len(rear_wps))
    )
    print(f"입력: {args.path_json}  (노드 {len(rear_wps)}개, 방향전환 {n_switch}회, 길이 {total_len:.1f}m)")
    print(f"단위: {args.units}  (스케일 x{scale:g}; 원본 road_model.dxf와 {'동일 좌표' if args.units=='mm' else '미터 좌표'})")
    print(f"구간(run) 수: {len(runs)}  ->", ", ".join(f"{g}:{j-k}노드" for (k, j, g) in runs))
    print("레이어:", ", ".join(LAYERS.keys()))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
