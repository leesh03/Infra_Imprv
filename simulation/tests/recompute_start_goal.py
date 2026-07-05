"""사용자 재정의 반영 (2026-07-04): 시작/목표 포즈를 회랑 안쪽 임의 지점이
아니라 **경사로 자체의 끝단 폭 선분**을 기준으로 다시 잡는다.

- 시작: 버스 뒷면이 "아래 경사로 시작점 선분"(폭 방향) 중앙에 닿는 순간의
  버스 앞면 중앙 점 위치를 기준으로 삼는다.
- 목표: 버스 앞면 중앙이 "위 경사로 끝 선분"(폭 방향) 중앙에 닿는 순간의
  위치를 기준으로 삼는다.

두 경사로의 "끝단 폭 선분"은 raw_plan.json에서 다른 긴 스플라인 조각들과
달리 **2점짜리 직선 세그먼트**로 남아있다 — DXF에서 경사로 단면을 닫는
캡(cap) 선이다. 자동으로 찾지 않고 좌표를 명시해 재현성을 보장한다."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from shapely.geometry import Point, Polygon

SIM_DIR = Path(__file__).resolve().parent.parent

# raw_plan.json에서 확인된 2점 세그먼트 (아래 경사로 시작단 / 위 경사로 끝단)
LOWER_RAMP_CAP = [(-24.17305115574085, -9.017464343150568), (-22.77128456857617, -13.27205353230813)]
UPPER_RAMP_CAP = [(-27.862738022661333, 4.6806951957591085), (-29.283931896383688, -1.602599451743372)]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _cap_normal_into_polygon(cap, boundary_polygon, probe=0.5):
    """cap(2점) 중앙에서 두 수직 방향 중 폴리곤 안쪽으로 들어가는 쪽을 고른다."""
    (x0, y0), (x1, y1) = cap
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    n1 = (-uy, ux)
    n2 = (uy, -ux)
    p1 = Point(mx + n1[0] * probe, my + n1[1] * probe)
    p2 = Point(mx + n2[0] * probe, my + n2[1] * probe)
    inward = n1 if boundary_polygon.contains(p1) else n2
    if boundary_polygon.contains(p1) and boundary_polygon.contains(p2):
        raise RuntimeError("both probe points inside boundary — cap orientation ambiguous, increase probe")
    if not boundary_polygon.contains(p1) and not boundary_polygon.contains(p2):
        raise RuntimeError("neither probe point inside boundary — check cap coordinates")
    return (mx, my), inward, length


def main():
    with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(SIM_DIR / "geometry" / "road_boundary.json", "r", encoding="utf-8") as f:
        boundary = Polygon(json.load(f)["polygon"])

    bus = cfg["bus"]
    length_total = bus["length_total"]
    d_ref = bus["d_ref"]  # 뒷축 -> 앞면 (현재 config 값 그대로 사용, 논의 중인 값)

    # --- 시작: 뒷면이 아래 경사로 시작단 중앙에 닿음 -> 그 순간 앞면중심 위치 ---
    rear_face_mid, into_dir, cap_w_start = _cap_normal_into_polygon(LOWER_RAMP_CAP, boundary)
    heading_start = math.atan2(into_dir[1], into_dir[0])
    front_center_start = (
        rear_face_mid[0] + length_total * into_dir[0],
        rear_face_mid[1] + length_total * into_dir[1],
    )
    rear_axle_start = (
        front_center_start[0] - d_ref * math.cos(heading_start),
        front_center_start[1] - d_ref * math.sin(heading_start),
    )

    # --- 목표: 앞면중심이 위 경사로 끝단 중앙에 닿음 -> 그 순간 앞면중심 위치 ---
    front_face_mid, into_dir_g, cap_w_goal = _cap_normal_into_polygon(UPPER_RAMP_CAP, boundary)
    # 목표 지점에서는 버스가 캡을 향해 "빠져나가는" 방향으로 향해 있어야 하므로,
    # 폴리곤 안쪽을 향하는 into_dir_g의 반대(바깥쪽)가 실제 진행 방향(heading).
    heading_goal = math.atan2(-into_dir_g[1], -into_dir_g[0])
    front_center_goal = front_face_mid
    rear_axle_goal = (
        front_center_goal[0] - d_ref * math.cos(heading_goal),
        front_center_goal[1] - d_ref * math.sin(heading_goal),
    )

    print("=== LOWER RAMP CAP (start) ===")
    print(f"  cap width={cap_w_start:.3f}m  rear_face_mid={rear_face_mid}")
    print(f"  heading={math.degrees(heading_start):.2f} deg")
    print(f"  front_center={front_center_start}")
    print(f"  rear_axle (new start pose)= x={rear_axle_start[0]:.3f} y={rear_axle_start[1]:.3f} "
          f"theta={heading_start:.4f} rad ({math.degrees(heading_start):.2f} deg)")

    print("=== UPPER RAMP CAP (goal) ===")
    print(f"  cap width={cap_w_goal:.3f}m  front_face_mid={front_face_mid}")
    print(f"  heading={math.degrees(heading_goal):.2f} deg")
    print(f"  rear_axle (new goal pose)= x={rear_axle_goal[0]:.3f} y={rear_axle_goal[1]:.3f} "
          f"theta={heading_goal:.4f} rad ({math.degrees(heading_goal):.2f} deg)")

    # --- 전체 OBB 충돌 확인 ---
    half_w = bus["width"] / 2.0
    front_oh = d_ref
    rear_oh = length_total - d_ref

    def bus_polygon(x, y, theta):
        c, s = math.cos(theta), math.sin(theta)
        local = [(front_oh, half_w), (front_oh, -half_w), (-rear_oh, -half_w), (-rear_oh, half_w)]
        world = [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]
        return Polygon(world)

    def violation_area(pose, theta):
        return bus_polygon(pose[0], pose[1], theta).difference(boundary).area

    def nudge_inward(rear_axle, theta, inward_unit, max_shift=3.0, step=0.02):
        """rear_axle을 inward_unit 방향으로 조금씩 밀어 위반 면적이 실질적으로
        0(<1e-6 m^2, 부동소수 오차 수준)이 될 때까지 전진. heading은 고정."""
        shift = 0.0
        pose = rear_axle
        while violation_area(pose, theta) > 1e-6 and shift < max_shift:
            shift += step
            pose = (rear_axle[0] + inward_unit[0] * shift, rear_axle[1] + inward_unit[1] * shift)
        return pose, shift

    v_start = violation_area(rear_axle_start, heading_start)
    v_goal = violation_area(rear_axle_goal, heading_goal)
    print(f"\nstart violation area: {v_start:.6f} m^2 (bus area {29.39:.1f} m^2 기준)")
    print(f"goal  violation area: {v_goal:.6f} m^2")

    if v_start > 1e-6:
        rear_axle_start, shift = nudge_inward(rear_axle_start, heading_start, into_dir)
        print(f"  -> start를 안쪽으로 {shift:.2f}m 밀어 충돌 해소")
    if v_goal > 1e-6:
        # goal은 into_dir_g가 "안쪽" 방향(코너/캡 쪽에서 apex 쪽으로)이므로 그대로 사용
        rear_axle_goal, shift = nudge_inward(rear_axle_goal, heading_goal, into_dir_g)
        print(f"  -> goal을 안쪽으로 {shift:.2f}m 밀어 충돌 해소")
        front_center_goal = (
            rear_axle_goal[0] + d_ref * math.cos(heading_goal),
            rear_axle_goal[1] + d_ref * math.sin(heading_goal),
        )

    start_poly = bus_polygon(*rear_axle_start, heading_start)
    goal_poly = bus_polygon(*rear_axle_goal, heading_goal)
    start_ok = boundary.contains(start_poly)
    goal_ok = boundary.contains(goal_poly)
    print(f"\nstart full-OBB collision-free (nudge 후): {start_ok}")
    print(f"goal  full-OBB collision-free (nudge 후): {goal_ok}")

    # 기존 config 값과 비교
    old_start = (cfg["start"]["x"], cfg["start"]["y"], cfg["start"]["heading"])
    old_goal_fc = (cfg["goal"]["x"], cfg["goal"]["y"], cfg["goal"]["heading"])
    print(f"\n기존 config start (rear axle): {old_start}")
    print(f"기존 config goal (front-center): {old_goal_fc}")
    dist_start = math.hypot(rear_axle_start[0] - old_start[0], rear_axle_start[1] - old_start[1])
    dist_goal = math.hypot(front_center_goal[0] - old_goal_fc[0], front_center_goal[1] - old_goal_fc[1])
    print(f"기존 대비 이동 거리: start {dist_start:.2f}m, goal {dist_goal:.2f}m")

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(10, 9))
    px, py = boundary.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)

    ox, oy = start_poly.exterior.xy
    ax.fill(ox, oy, color="tab:green", alpha=0.4, label="new start (rear touches ramp cap)")
    gx, gy = goal_poly.exterior.xy
    ax.fill(gx, gy, color="tab:purple", alpha=0.4, label="new goal (front touches ramp cap)")

    old_start_poly = bus_polygon(old_start[0], old_start[1], old_start[2])
    ox2, oy2 = old_start_poly.exterior.xy
    ax.plot(ox2, oy2, "g--", linewidth=1, label="old start (config)")

    old_goal_rear_x = old_goal_fc[0] - d_ref * math.cos(old_goal_fc[2])
    old_goal_rear_y = old_goal_fc[1] - d_ref * math.sin(old_goal_fc[2])
    old_goal_poly = bus_polygon(old_goal_rear_x, old_goal_rear_y, old_goal_fc[2])
    ox3, oy3 = old_goal_poly.exterior.xy
    ax.plot(ox3, oy3, "m--", linewidth=1, label="old goal (config)")

    capx0, capy0 = zip(*LOWER_RAMP_CAP)
    ax.plot(capx0, capy0, "b-", linewidth=3, label="lower ramp start cap")
    capx1, capy1 = zip(*UPPER_RAMP_CAP)
    ax.plot(capx1, capy1, "r-", linewidth=3, label="upper ramp end cap")

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Start/Goal pose redefinition: ramp-cap based vs old config")
    fig.tight_layout()
    fig.savefig(SIM_DIR / "outputs" / "start_goal_redefinition.png", dpi=140)
    print("\nsaved outputs/start_goal_redefinition.png")

    return {
        "start": {"x": rear_axle_start[0], "y": rear_axle_start[1], "heading": heading_start},
        "goal_front_center": {"x": front_center_goal[0], "y": front_center_goal[1], "heading": heading_goal},
        "start_ok": start_ok, "goal_ok": goal_ok,
    }


if __name__ == "__main__":
    main()
