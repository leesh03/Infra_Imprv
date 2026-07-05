"""새 경계(v3 dxf)에서 실제로 주행 가능한(전/후진 포함) 기준 경로를 시뮬레이션으로
직접 생성한다. 매 스텝 실제 버스 OBB가 경계를 벗어나지 않는지 검증하면서
진행하므로, 손으로 그린 중심선보다 신뢰도가 높다.

단계: NAV_IN(진입 회랑 전진) -> ALIGN(루프에서 전후진 반복 제자리 회전) ->
      NAV_OUT(진출 회랑 전진) -> FINAL(목표 정밀 접근)
각 스텝의 front_center 좌표 + gear(FWD/REV)를 기록해 reference_path.json으로 저장한다.
"""
import json
import math
from pathlib import Path

import numpy as np

from envs.bus_hairpin_env import BusHairpinEnv, _wrap_angle
from kinematics.bicycle import BicycleState

SIM_DIR = Path(__file__).resolve().parent.parent


def pose_error_to(env, state, tx, ty, theading):
    fc_x, fc_y = env.front_center(state)
    dx, dy = tx - fc_x, ty - fc_y
    dtheta = _wrap_angle(theading - state.theta)
    c, s = math.cos(state.theta), math.sin(state.theta)
    dx_body = dx * c + dy * s
    dy_body = -dx * s + dy * c
    return dx_body, dy_body, dtheta, math.hypot(dx, dy)


def nav_action(env, state, tx, ty, theading, speed_frac=0.5):
    dx_body, dy_body, dtheta, pos_err = pose_error_to(env, state, tx, ty, theading)
    v, delta = state.v, state.delta
    alpha = math.atan2(dy_body, dx_body)
    gear = 1.0 if math.cos(alpha) >= 0 else -1.0
    speed_cap = env.model.v_max * speed_frac if pos_err > 3.0 else max(0.1, pos_err * 0.3)
    target_v = gear * speed_cap
    a_accel = np.clip((target_v - v) * 2.0, -1.0, 1.0)
    alpha_eff = alpha if gear > 0 else _wrap_angle(alpha - math.pi)
    heading_eff = dtheta * gear
    w_heading = np.clip(1.0 - pos_err / 4.0, 0.0, 1.0)
    combined = (1.0 - w_heading) * alpha_eff + w_heading * heading_eff
    desired_delta = np.clip(combined * 1.5, -env.model.delta_max, env.model.delta_max)
    a_steer = np.clip((desired_delta - delta) * 3.0, -1.0, 1.0)
    return np.array([a_accel, a_steer], dtype=np.float32), pos_err


def final_action(env, state):
    dx_body, dy_body, dtheta, pos_err = pose_error_to(env, state, env.goal_x, env.goal_y, env.goal_heading)
    alpha = math.atan2(dy_body, dx_body)
    gear = 1.0 if math.cos(alpha) >= 0 else -1.0
    target_v = gear * max(0.1, pos_err * 0.3)
    a_accel = np.clip((target_v - state.v) * 2.0, -1.0, 1.0)
    alpha_eff = alpha if gear > 0 else _wrap_angle(alpha - math.pi)
    heading_eff = dtheta * gear
    w_heading = np.clip(1.0 - pos_err / 4.0, 0.0, 1.0)
    combined = (1.0 - w_heading) * alpha_eff + w_heading * heading_eff
    desired_delta = np.clip(combined * 1.5, -env.model.delta_max, env.model.delta_max)
    a_steer = np.clip((desired_delta - state.delta) * 3.0, -1.0, 1.0)
    return np.array([a_accel, a_steer], dtype=np.float32), pos_err, dtheta


def run_plan_shuttle(staging_xy, entry_heading, exit_heading, n_subtargets, arm_m, max_steps=4000):
    """ALIGN을 '제자리 최대조향 반복'이 아니라, staging 주변에 짧은 팔길이
    sub-target을 두고 그 지점으로 번갈아 pure-pursuit(nav_action)하는 방식으로
    구현. nav_action은 위치+헤딩을 함께 추종하므로 매 구간이 자기교정되어
    드리프트가 누적되지 않는다."""
    env = BusHairpinEnv()
    state = BicycleState(x=env.start_x, y=env.start_y, theta=env.start_heading, v=0.0, delta=0.0)

    loop_centerline = [(2.0, 0.48), (4.0, 0.70), (6.0, 0.87), (8.0, 0.88), (10.0, 0.77),
                       (12.0, 0.14), (14.0, -1.64), (16.0, -3.30), (18.0, -2.97), (20.0, -2.08)]
    loop_leg = [p for p in loop_centerline if p[0] <= staging_xy[0] + 1e-6]
    nav_in = [(-8.0, -5.62), (-2.0, -3.63), (0.0, -3.06)] + loop_leg
    if nav_in[-1] != staging_xy:
        nav_in.append(staging_xy)
    nav_out = [(-2.0, 2.95), (-6.0, 2.71), (-10.0, 2.02), (-14.0, 1.49), (-18.0, 1.16)]

    headings = np.linspace(entry_heading, entry_heading + _wrap_angle(exit_heading - entry_heading), n_subtargets + 1)[1:]
    sub_targets = [(staging_xy[0] + arm_m * math.cos(h), staging_xy[1] + arm_m * math.sin(h), h) for h in headings]

    phase = "NAV_IN"
    wp_idx = 0
    sub_idx = 0
    traj = []

    def record(state_, gear):
        fc = env.front_center(state_)
        traj.append((fc[0], fc[1], gear, phase))

    record(state, "FWD")
    for t in range(max_steps):
        if phase == "NAV_IN":
            tx, ty = nav_in[wp_idx]
            if wp_idx < len(nav_in) - 1:
                nx, ny = nav_in[wp_idx + 1]
                theading = math.atan2(ny - ty, nx - tx)
            else:
                theading = entry_heading
            action, pos_err = nav_action(env, state, tx, ty, theading)
            if pos_err < 2.0:
                if wp_idx < len(nav_in) - 1:
                    wp_idx += 1
                else:
                    phase = "ALIGN"
        elif phase == "ALIGN":
            tx, ty, theading = sub_targets[sub_idx]
            action, pos_err = nav_action(env, state, tx, ty, theading, speed_frac=0.3)
            dtheta_final = _wrap_angle(exit_heading - state.theta)
            if pos_err < 1.5 or abs(_wrap_angle(theading - state.theta)) < math.radians(12):
                if sub_idx < len(sub_targets) - 1:
                    sub_idx += 1
                else:
                    phase = "NAV_OUT"
                    wp_idx = 0
        elif phase == "NAV_OUT":
            tx, ty = nav_out[wp_idx]
            if wp_idx < len(nav_out) - 1:
                nx, ny = nav_out[wp_idx + 1]
                theading = math.atan2(ny - ty, nx - tx)
            else:
                theading = env.goal_heading
            action, pos_err = nav_action(env, state, tx, ty, theading)
            if pos_err < 2.5:
                if wp_idx < len(nav_out) - 1:
                    wp_idx += 1
                else:
                    phase = "FINAL"
        else:
            action, pos_err, dtheta_f = final_action(env, state)

        grade_here = env.grade_field.grade_at(state.x, state.y)
        next_state = env.model.step(state, float(action[0]) * env.a_max, float(action[1]) * env.steer_rate_max, grade=grade_here)
        gear = "FWD" if next_state.v >= 0 else "REV"
        collided = env._check_collision(next_state)
        state = next_state
        record(state, gear)
        if collided:
            return {"success": False, "reason": "collision", "step": t, "phase": phase, "traj": traj, "env": env}
        if phase == "FINAL" and env._is_success(state):
            return {"success": True, "step": t, "traj": traj, "env": env}
    return {"success": False, "reason": "timeout", "phase": phase, "traj": traj, "env": env}


def run_plan(staging_xy, switch_steps, align_speed_frac, max_steps=4000, verbose=False):
    env = BusHairpinEnv()
    state = BicycleState(x=env.start_x, y=env.start_y, theta=env.start_heading, v=0.0, delta=0.0)

    loop_centerline = [(2.0, 0.48), (4.0, 0.70), (6.0, 0.87), (8.0, 0.88), (10.0, 0.77),
                       (12.0, 0.14), (14.0, -1.64), (16.0, -3.30), (18.0, -2.97), (20.0, -2.08)]
    loop_leg = [p for p in loop_centerline if p[0] <= staging_xy[0] + 1e-6]
    nav_in = [(-8.0, -5.62), (-2.0, -3.63), (0.0, -3.06)] + loop_leg
    if nav_in[-1] != staging_xy:
        nav_in.append(staging_xy)
    nav_out = [(-2.0, 2.95), (-6.0, 2.71), (-10.0, 2.02), (-14.0, 1.49), (-18.0, 1.16)]

    # ALIGN 목표 헤딩: staging 지점에서 nav_out 첫 목표를 바라보는 방향
    tgt_heading = math.atan2(nav_out[0][1] - staging_xy[1], nav_out[0][0] - staging_xy[0])

    phase = "NAV_IN"
    wp_idx = 0
    align_step = 0
    traj = []  # (fc_x, fc_y, gear, phase, collision)

    def record(state_, gear):
        fc = env.front_center(state_)
        traj.append((fc[0], fc[1], gear, phase))

    record(state, "FWD")
    for t in range(max_steps):
        if phase == "NAV_IN":
            tx, ty = nav_in[wp_idx]
            if wp_idx < len(nav_in) - 1:
                nx, ny = nav_in[wp_idx + 1]
                theading = math.atan2(ny - ty, nx - tx)
            else:
                theading = tgt_heading
            action, pos_err = nav_action(env, state, tx, ty, theading)
            if pos_err < 2.5:
                if wp_idx < len(nav_in) - 1:
                    wp_idx += 1
                else:
                    phase = "ALIGN"
        elif phase == "ALIGN":
            dtheta = _wrap_angle(tgt_heading - state.theta)
            gear_forward = (align_step // switch_steps) % 2 == 0
            sign = 1.0 if dtheta > 0 else -1.0
            # 순수 전후진 반전만으로는 매 사이클 회전축이 조금씩 밀려 누적
            # 드리프트가 생겨 결국 벽에 닿는다 (실측: staging 근처에서 시작해도
            # 수백 스텝 뒤 y가 수 m 밀림). staging 지점으로 되돌리는 위치
            # 피드백 항을 조향에 더해 회전축을 고정한다(제자리 회전에 가깝게).
            pivot_dx = state.x - staging_xy[0]
            pivot_dy = state.y - staging_xy[1]
            lateral = -math.sin(state.theta) * pivot_dx + math.cos(state.theta) * pivot_dy
            recenter = np.clip(-lateral * 0.35, -0.5, 0.5)
            desired_delta = env.model.delta_max * sign if gear_forward else -env.model.delta_max * sign
            desired_delta = np.clip(desired_delta + recenter, -env.model.delta_max, env.model.delta_max)
            target_v = (env.model.v_max if gear_forward else env.model.v_min) * align_speed_frac
            a_accel = np.clip((target_v - state.v) * 5.0, -1.0, 1.0)
            a_steer = np.clip((desired_delta - state.delta) * 5.0, -1.0, 1.0)
            action = np.array([a_accel, a_steer], dtype=np.float32)
            align_step += 1
            if abs(dtheta) < math.radians(10.0):
                phase = "NAV_OUT"
                wp_idx = 0
        elif phase == "NAV_OUT":
            tx, ty = nav_out[wp_idx]
            if wp_idx < len(nav_out) - 1:
                nx, ny = nav_out[wp_idx + 1]
                theading = math.atan2(ny - ty, nx - tx)
            else:
                theading = env.goal_heading
            action, pos_err = nav_action(env, state, tx, ty, theading)
            if pos_err < 2.5:
                if wp_idx < len(nav_out) - 1:
                    wp_idx += 1
                else:
                    phase = "FINAL"
        else:  # FINAL
            action, pos_err, dtheta_f = final_action(env, state)

        grade_here = env.grade_field.grade_at(state.x, state.y)
        next_state = env.model.step(state, float(action[0]) * env.a_max, float(action[1]) * env.steer_rate_max, grade=grade_here)
        gear = "FWD" if next_state.v >= 0 else "REV"

        collided = env._check_collision(next_state)
        state = next_state
        record(state, gear)
        if collided:
            return {"success": False, "reason": "collision", "step": t, "phase": phase, "traj": traj, "env": env}

        if phase == "FINAL" and env._is_success(state):
            return {"success": True, "step": t, "traj": traj, "env": env}

    return {"success": False, "reason": "timeout", "phase": phase, "traj": traj, "env": env}


if __name__ == "__main__":
    candidates = [
        (10.0, 0.77), (12.0, 0.14), (14.0, -1.64), (16.0, -3.30), (18.0, -2.97), (20.0, -2.08),
    ]
    switch_opts = [15, 20, 25, 30, 45]
    speed_opts = [0.2, 0.3, 0.4]

    best = None
    for staging in candidates:
        for switch in switch_opts:
            for speed in speed_opts:
                res = run_plan(staging, switch, speed)
                tag = f"staging={staging} switch={switch} speed={speed}"
                if res["success"]:
                    dist = 0.0
                    traj = res["traj"]
                    for i in range(1, len(traj)):
                        dist += math.hypot(traj[i][0] - traj[i - 1][0], traj[i][1] - traj[i - 1][1])
                    print(f"OK  {tag}  steps={res['step']} path_len={dist:.1f}")
                    if best is None or dist < best[0]:
                        best = (dist, staging, switch, speed, res)
                else:
                    print(f"FAIL {tag}  reason={res['reason']} phase={res.get('phase')} step={res.get('step')}")
    if best:
        print("BEST:", best[1], best[2], best[3], "path_len=", best[0])
    else:
        print("NO FEASIBLE PLAN FOUND")
