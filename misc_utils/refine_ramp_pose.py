"""recompute_start_goal.py에서 확인된 문제: 위 경사로 끝단 바로 안쪽에서
회랑이 휘어 있어, heading을 캡 수직 방향으로 고정한 채 안쪽으로 미는 것만으로는
충돌이 오히려 커진다. 여기서는 "이상적 지점(캡 중앙, 캡 수직 heading)"에서
가장 가까운 충돌없는 (전진거리 d, 헤딩보정 dtheta, 캡을 따른 옆이동 s) 조합을
그리드 탐색으로 찾는다. start도 동일한 틀로 재확인(0,0,0 근처에서 이미 충돌
없음을 재검증)."""
import json
import math
from pathlib import Path

import yaml
from shapely.geometry import Polygon

from tests.recompute_start_goal import LOWER_RAMP_CAP, UPPER_RAMP_CAP, _cap_normal_into_polygon

SIM_DIR = Path(__file__).resolve().parent.parent


def _bus_polygon(x, y, theta, front_oh, rear_oh, half_w):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front_oh, half_w), (front_oh, -half_w), (-rear_oh, -half_w), (-rear_oh, half_w)]
    world = [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]
    return Polygon(world)


def search_best_pose(cap_mid, ideal_heading, along_dir, tangent_dir, boundary,
                      front_oh, rear_oh, half_w, ref_is_front, cap_half_width,
                      d_range, dtheta_deg_range, s_range, safety_boundary=None):
    """ref_is_front=True면 (cap_mid + s*tangent + d*along)이 앞면중심,
    False면 뒷면중심이 되도록 rear_axle을 역산한다.

    safety_boundary: 실제 경계 대신 이 폴리곤(보통 boundary를 살짝
    안쪽으로 buffer한 것) 기준으로 충돌 여부를 판정한다 — 정확히 벽에
    닿기 직전인 지점을 고르면 그 지점에서 뻗어나갈 수 있는 조향 여지가
    거의 없어(첫 스텝 대부분이 바로 충돌) Hybrid A* 탐색 자체가 막히는
    문제가 있어, 약간의 여유(margin)를 갖는 지점을 우선한다."""
    check_boundary = safety_boundary if safety_boundary is not None else boundary
    best = None  # (cost, d, dtheta_deg, s, rear_axle, heading)
    for d in d_range:
        for dtheta_deg in dtheta_deg_range:
            heading = ideal_heading + math.radians(dtheta_deg)
            c, s_ = math.cos(heading), math.sin(heading)
            for s in s_range:
                ref_x = cap_mid[0] + tangent_dir[0] * s + along_dir[0] * d
                ref_y = cap_mid[1] + tangent_dir[1] * s + along_dir[1] * d
                if ref_is_front:
                    rear_axle = (ref_x - front_oh * c, ref_y - front_oh * s_)
                else:
                    rear_axle = (ref_x + rear_oh * c, ref_y + rear_oh * s_)
                poly = _bus_polygon(rear_axle[0], rear_axle[1], heading, front_oh, rear_oh, half_w)
                viol = poly.difference(check_boundary).area
                if viol > 1e-6:
                    continue
                cost = d ** 2 + (dtheta_deg / 10.0) ** 2 + (s / max(cap_half_width, 0.1)) ** 2
                if best is None or cost < best[0]:
                    best = (cost, d, dtheta_deg, s, rear_axle, heading)
    return best


def main():
    with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(SIM_DIR / "geometry" / "road_boundary.json", "r", encoding="utf-8") as f:
        boundary = Polygon(json.load(f)["polygon"])

    bus = cfg["bus"]
    length_total = bus["length_total"]
    d_ref = bus["d_ref"]
    half_w = bus["width"] / 2.0
    front_oh = d_ref
    rear_oh = length_total - d_ref

    d_range = [i * 0.25 for i in range(0, 41)]  # 0 ~ 10m
    dtheta_range = list(range(-45, 46, 2))
    s_range = [0.0]  # 1차: 좌우이동 없이
    SAFETY_MARGIN_M = 0.15
    safety_boundary = boundary.buffer(-SAFETY_MARGIN_M)

    # --- START: 뒷면중심 = cap 중앙 + s*tangent + d*(안쪽 방향) ---
    rear_face_mid, into_dir, cap_w0 = _cap_normal_into_polygon(LOWER_RAMP_CAP, boundary)
    tangent0 = (-into_dir[1], into_dir[0])
    ideal_heading_start = math.atan2(into_dir[1], into_dir[0])
    s_range0 = [i * 0.2 for i in range(-8, 9)]
    best_start = search_best_pose(
        rear_face_mid, ideal_heading_start, into_dir, tangent0, boundary,
        front_oh, rear_oh, half_w, ref_is_front=False, cap_half_width=cap_w0 / 2,
        d_range=d_range, dtheta_deg_range=dtheta_range, s_range=s_range0,
        safety_boundary=safety_boundary,
    )
    print("START best:", best_start[:4] if best_start else None)
    if best_start:
        _, d, dth, s, rear_axle, heading = best_start
        print(f"  d(전진)={d:.2f}m dtheta={dth}deg s(옆이동)={s:.2f}m")
        print(f"  rear_axle=({rear_axle[0]:.3f},{rear_axle[1]:.3f}) heading={heading:.4f}rad({math.degrees(heading):.2f}deg)")

    # --- GOAL: 앞면중심 = cap 중앙 + s*tangent + d*(안쪽 방향) ---
    front_face_mid, into_dir_g, cap_w1 = _cap_normal_into_polygon(UPPER_RAMP_CAP, boundary)
    tangent1 = (-into_dir_g[1], into_dir_g[0])
    ideal_heading_goal = math.atan2(-into_dir_g[1], -into_dir_g[0])
    s_range1 = [i * 0.3 for i in range(-10, 11)]
    best_goal = search_best_pose(
        front_face_mid, ideal_heading_goal, into_dir_g, tangent1, boundary,
        front_oh, rear_oh, half_w, ref_is_front=True, cap_half_width=cap_w1 / 2,
        d_range=d_range, dtheta_deg_range=dtheta_range, s_range=s_range1,
        safety_boundary=safety_boundary,
    )
    print("GOAL best:", best_goal[:4] if best_goal else None)
    if best_goal:
        _, d, dth, s, rear_axle, heading = best_goal
        print(f"  d(전진)={d:.2f}m dtheta={dth}deg s(옆이동)={s:.2f}m")
        print(f"  rear_axle=({rear_axle[0]:.3f},{rear_axle[1]:.3f}) heading={heading:.4f}rad({math.degrees(heading):.2f}deg)")
        fc = (rear_axle[0] + front_oh * math.cos(heading), rear_axle[1] + front_oh * math.sin(heading))
        print(f"  front_center=({fc[0]:.3f},{fc[1]:.3f})")

    return best_start, best_goal


if __name__ == "__main__":
    main()
