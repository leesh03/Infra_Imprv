"""전/후진을 포함한 기준 경로를 키프레임(앞면중심 x,y,heading,gear)으로 정의하고,
키프레임 사이를 촘촘히 보간(위치 선형, heading 최단각 보간)해 매 샘플에서 버스
전체 OBB(뒷축 역산)가 경계 폴리곤 안에 있는지 검사한다.

주의: 보간은 실제 bicycle 동역학이 아니라 선형/각도 보간 근사이므로 완전한
운동학적 curvature 보증은 아니지만, 각 키프레임 자체는 정확히 검증되고 그
사이 구간도 촘촘한 샘플로 확인하므로 명백한 경계 이탈은 확실히 잡아낸다.

후진 구간: 후진 중에는 앞면중심이 heading의 "반대" 방향으로 움직인다
(뒷축이 -heading 방향으로 이동하므로). 그래서 REV 구간 키프레임의 heading은
위치 시퀀스의 접선 + 180도로 계산한다.
"""
import json
import math

from envs.bus_hairpin_env import BusHairpinEnv, _wrap_angle
from kinematics.bicycle import BicycleState


def interpolate(keyframes, samples_per_seg=25):
    out = []
    for i in range(len(keyframes) - 1):
        x0, y0, th0, _ = keyframes[i]
        x1, y1, th1, gear1 = keyframes[i + 1]
        dth = _wrap_angle(th1 - th0)
        last = (i == len(keyframes) - 2)
        n = samples_per_seg + (1 if last else 0)
        for k in range(n):
            f = k / samples_per_seg
            x = x0 + (x1 - x0) * f
            y = y0 + (y1 - y0) * f
            th = th0 + dth * f
            out.append((x, y, th, gear1))
    return out


def verify(env, front_center_samples):
    violations = []
    for i, (fx, fy, th, gear) in enumerate(front_center_samples):
        rax_x = fx - env.d_ref * math.cos(th)
        rax_y = fy - env.d_ref * math.sin(th)
        st = BicycleState(x=rax_x, y=rax_y, theta=th, v=0.0, delta=0.0)
        bus_poly = env._bus_polygon(st)
        if not env.boundary_polygon.contains(bus_poly):
            clearance = env.boundary_polygon.exterior.distance(bus_poly)
            violations.append((i, fx, fy, th, gear, clearance))
    return violations


def auto_headings(positions, gears, start_heading, end_heading):
    n = len(positions)
    headings = [start_heading]
    for i in range(1, n - 1):
        x0, y0 = positions[i - 1]
        x1, y1 = positions[i + 1]
        tangent = math.atan2(y1 - y0, x1 - x0)
        if gears[i] == "REV":
            tangent = _wrap_angle(tangent + math.pi)
        headings.append(tangent)
    headings.append(end_heading)
    return headings


if __name__ == "__main__":
    env = BusHairpinEnv()
    start_fc = env.front_center(BicycleState(x=env.start_x, y=env.start_y, theta=env.start_heading, v=0, delta=0))

    print("start_fc", start_fc)
    positions = [
        (start_fc[0], start_fc[1]),
        (-8.0, -5.62), (-4.0, -4.18), (-2.0, -3.63), (-1.0, -3.3), (0.0, -2.8),
        (0.5, -2.0), (1.0, -1.0), (1.4, 0.0), (1.7, 1.0), (2.0, 1.8), (2.5, 2.6), (5.0, 3.0),  # 전진: 루프로 진입, 코를 들이민 지점
        (2.0, 2.6), (0.0, 3.16),           # 후진: 빠지면서 헤딩을 진출 방향으로
        (-2.0, 2.95), (-6.0, 2.71), (-10.0, 2.02), (-14.0, 1.49), (-18.0, 1.16),
        (env.goal_x, env.goal_y),
    ]
    gears = ["FWD"] * len(positions)
    gears[13] = "REV"
    gears[14] = "REV"

    headings = auto_headings(positions, gears, env.start_heading, env.goal_heading)
    keyframes = [(p[0], p[1], h, g) for p, h, g in zip(positions, headings, gears)]

    samples = interpolate(keyframes, samples_per_seg=20)
    violations = verify(env, samples)
    print(f"samples={len(samples)} violations={len(violations)}")
    for v in violations[:30]:
        print("  idx=%d fc=(%.2f,%.2f) th=%.2f gear=%s clearance=%.3f" % v)

    if not violations:
        out = {
            "waypoints": [[round(k[0], 3), round(k[1], 3)] for k in keyframes],
            "gears": [k[3] for k in keyframes],
        }
        with open("geometry/reference_path.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("saved geometry/reference_path.json")
