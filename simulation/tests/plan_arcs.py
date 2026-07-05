"""정확한 원호 기하로 K-포인트 회차(전후진 반복) 궤적을 직접 계산한다.
(휴리스틱 제어기로 시행착오하는 대신, 최대조향(R_min) 원호를 정확히 이어붙여
계산하므로 경계 충돌 여부를 명확하게 검증할 수 있다.)

각 아크: gear=+1(전진+좌측 풀락) 또는 -1(후진+우측 풀락). 두 경우 모두 theta는
증가(CCW) 방향으로만 진행하도록 부호를 맞춰뒀다 (반대 회전방향을 원하면
좌우 락을 통째로 뒤집어 호출한다 - rotate_sign 인자).
"""
import math

from envs.bus_hairpin_env import BusHairpinEnv, _wrap_angle
from kinematics.bicycle import BicycleState


def arc_step(x, y, theta, gear, phi, R, rotate_sign=1.0):
    """gear=+1: 전진+좌락(rotate_sign=1이면 CCW), gear=-1: 후진+우락.
    rotate_sign=-1이면 전체 회전방향을 반대로(CW) 뒤집는다(좌우 락 반전)."""
    s = rotate_sign * (1.0 if gear > 0 else -1.0)  # ICR가 왼쪽/오른쪽 어느 쪽인지
    # ICR = pos + R*(-sin theta, cos theta)*s  (s=+1: 왼쪽 락 => CCW)
    icr_x = x + R * (-math.sin(theta)) * s
    icr_y = y + R * (math.cos(theta)) * s
    base_angle = theta - s * math.pi / 2.0
    dphi = rotate_sign * phi  # 항상 |theta| 증가량은 phi, 부호는 rotate_sign
    new_angle = base_angle + dphi
    nx = icr_x + R * math.cos(new_angle)
    ny = icr_y + R * math.sin(new_angle)
    ntheta = theta + dphi
    return nx, ny, ntheta


def sample_arc(x, y, theta, gear, phi, R, rotate_sign=1.0, n=12):
    pts = []
    for i in range(1, n + 1):
        f = phi * i / n
        px, py, pth = arc_step(x, y, theta, gear, f, R, rotate_sign)
        pts.append((px, py, pth))
    return pts


def build_trajectory(start_pose, arc_plan, R, rotate_sign=1.0):
    """arc_plan: [(gear, phi_deg), ...]. 반환: [(x,y,theta,gear), ...] (start 포함, 조밀 샘플)."""
    x, y, theta = start_pose
    traj = [(x, y, theta, "FWD")]
    for gear, phi_deg in arc_plan:
        phi = math.radians(phi_deg)
        pts = sample_arc(x, y, theta, gear, phi, R, rotate_sign)
        tag = "FWD" if gear > 0 else "REV"
        for px, py, pth in pts:
            traj.append((px, py, pth, tag))
        x, y, theta = pts[-1]
    return traj


def check_traj_collision(env, traj):
    for x, y, theta, _ in traj:
        st = BicycleState(x=x, y=y, theta=theta, v=0.0, delta=0.0)
        if env._check_collision(st):
            return True, (x, y, theta)
    return False, None


def search(env, start_pose, tgt_heading, R):
    dtheta_needed = _wrap_angle(tgt_heading - start_pose[2])
    results = []
    for rotate_sign in (1.0, -1.0):
        # rotate_sign이 필요한 회전 방향과 맞는지: 총 회전량 = rotate_sign이 만드는 방향의 dtheta_needed(같은 부호로 wrap)
        total_needed = dtheta_needed if (dtheta_needed * rotate_sign > 0) else dtheta_needed + rotate_sign * 2 * math.pi
        total_needed = abs(total_needed)
        for phi_deg in (20, 30, 40, 50, 60):
            phi = math.radians(phi_deg)
            n_half_arcs = math.ceil(total_needed / phi)
            if n_half_arcs < 2:
                n_half_arcs = 2
            # 마지막 아크는 남은 각도만큼만
            plan = []
            remaining = total_needed
            gear = 1
            while remaining > 1e-6:
                step = min(phi, remaining)
                plan.append((gear, math.degrees(step)))
                remaining -= step
                gear *= -1
            traj = build_trajectory(start_pose, plan, R, rotate_sign)
            collided, where = check_traj_collision(env, traj)
            end = traj[-1]
            results.append({
                "rotate_sign": rotate_sign, "phi_deg": phi_deg, "n_arcs": len(plan),
                "collided": collided, "where": where, "end": end, "traj": traj, "plan": plan,
            })
    return results


if __name__ == "__main__":
    env = BusHairpinEnv()
    R = env.model.R_min

    nav_out_first = (-2.0, 2.95)
    for staging in [(6.0, 0.87), (8.0, 0.88), (10.0, 0.77), (12.0, 0.14), (14.0, -1.64)]:
        tgt_heading = math.atan2(nav_out_first[1] - staging[1], nav_out_first[0] - staging[0])
        # 진입 헤딩은 apex->staging 방향으로 근사
        entry_heading = math.atan2(staging[1] - (-3.06), staging[0] - 0.0)
        start_pose = (staging[0], staging[1], entry_heading)
        results = search(env, start_pose, tgt_heading, R)
        ok = [r for r in results if not r["collided"]]
        print(f"staging={staging} entry_h={math.degrees(entry_heading):.1f} tgt_h={math.degrees(tgt_heading):.1f} "
              f"-> {len(ok)}/{len(results)} feasible")
        for r in sorted(ok, key=lambda r: r["n_arcs"])[:3]:
            print("   OK rotate_sign=%d phi=%d n_arcs=%d" % (r["rotate_sign"], r["phi_deg"], r["n_arcs"]))
