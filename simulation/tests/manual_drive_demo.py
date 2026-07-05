"""TASK 6 — 수동 조작 대체용 휴리스틱 컨트롤러로 목표 도달 가능성 검증.

3단계 전략 (실제 사람의 운전 방식을 모사):
  1) NAV   — 경계에서 충분히 떨어진 경유점들을 따라 정점 부근까지 이동
  2) ALIGN — 목표 헤딩과 큰 차이가 나면, 전진/후진을 번갈아(3점 회차) 헤딩만 보정
             (한 번의 호(arc)로는 R_min 제약상 좁은 공간에서 큰 헤딩 변화를 만들
             수 없어 전후진이 필요함 — 이 프로젝트의 핵심 문제 그 자체)
  3) FINAL — 헤딩이 맞춰지면 목표를 향해 직진하며 감속·정차

완전히 못 풀면 보상/제약 설계가 잘못된 것이라는 TASK6 기준에 따라, 환경의
충돌검사·동역학·목표판정이 실제로 도달 가능한 경로를 막지 않는지 확인한다.
"""
import math

import numpy as np

from envs.bus_hairpin_env import BusHairpinEnv, _wrap_angle

# v3 dxf 재작업(2026-07-02): geometry/reference_path.json 중심선 경유점을
# 사용 (첫 점=start는 env.reset() 시작 자세와 같으므로 제외). nav_action은
# front_center 기준으로 목표까지 방향(alpha)을 계산하므로, 다음 목표가
# d_ref(전방 오버행, 9.06m)보다 가까우면 "목표가 이미 뒤에 있다"고 오판해
# 출발하자마자 후진해버린다 — 그래서 촘촘한 원 경로점 중 연속점 간 최소
# MIN_SPACING_M 이상 떨어진 점만 남겨 성긴 NAV 경유점을 만든다.
MIN_SPACING_M = 6.0


def _load_nav_path():
    import json
    import math
    from pathlib import Path

    geometry_dir = Path(__file__).resolve().parent.parent / "geometry"
    with open(geometry_dir / "reference_path.json", "r", encoding="utf-8") as f:
        waypoints = [tuple(p) for p in json.load(f)["waypoints"]]

    thinned = []
    last = waypoints[0]  # start pose, not itself a nav target
    for p in waypoints[1:]:
        if math.hypot(p[0] - last[0], p[1] - last[1]) >= MIN_SPACING_M:
            thinned.append(p)
            last = p
    if not thinned or thinned[-1] != waypoints[-1]:
        thinned.append(waypoints[-1])
    return thinned


NAV_PATH = _load_nav_path()
NAV_CAPTURE_R = 3.0
ALIGN_TOL_DEG = 8.0
ALIGN_SWITCH_STEPS = 40   # 3점 회차 전후진 1구간 길이 (steps)
ALIGN_SPEED_FRAC = 0.3


def _pose_error_to(env, state, tx, ty, theading):
    fc_x, fc_y = env.front_center(state)
    dx, dy = tx - fc_x, ty - fc_y
    dtheta = _wrap_angle(theading - state.theta)
    c, s = math.cos(state.theta), math.sin(state.theta)
    dx_body = dx * c + dy * s
    dy_body = -dx * s + dy * c
    return dx_body, dy_body, dtheta, math.hypot(dx, dy)


def nav_action(env, state, tx, ty, theading):
    dx_body, dy_body, dtheta, pos_err = _pose_error_to(env, state, tx, ty, theading)
    v, delta = state.v, state.delta

    alpha = math.atan2(dy_body, dx_body)
    gear = 1.0 if math.cos(alpha) >= 0 else -1.0

    speed_cap = env.model.v_max * 0.5 if pos_err > 3.0 else max(0.1, pos_err * 0.3)
    target_v = gear * speed_cap
    a_accel = np.clip((target_v - v) * 2.0, -1.0, 1.0)

    alpha_eff = alpha if gear > 0 else _wrap_angle(alpha - math.pi)
    heading_eff = dtheta * gear
    w_heading = np.clip(1.0 - pos_err / 4.0, 0.0, 1.0)
    combined = (1.0 - w_heading) * alpha_eff + w_heading * heading_eff

    desired_delta = np.clip(combined * 1.5, -env.model.delta_max, env.model.delta_max)
    a_steer = np.clip((desired_delta - delta) * 3.0, -1.0, 1.0)
    return np.array([a_accel, a_steer], dtype=np.float32), pos_err


def align_action(env, state, goal_heading, step_in_phase):
    dtheta = _wrap_angle(goal_heading - state.theta)
    gear_forward = (step_in_phase // ALIGN_SWITCH_STEPS) % 2 == 0
    sign = 1.0 if dtheta > 0 else -1.0
    desired_delta = env.model.delta_max * sign if gear_forward else -env.model.delta_max * sign
    target_v = (env.model.v_max if gear_forward else env.model.v_min) * ALIGN_SPEED_FRAC

    a_accel = np.clip((target_v - state.v) * 5.0, -1.0, 1.0)
    a_steer = np.clip((desired_delta - state.delta) * 5.0, -1.0, 1.0)
    return np.array([a_accel, a_steer], dtype=np.float32), dtheta


def final_action(env, state):
    """목표 지점을 향한 pure-pursuit(alpha)과 목표 헤딩 정렬(dtheta)을 거리에
    따라 블렌드. (단순히 dtheta만 쓰면 헤딩은 맞아도 옆으로 비껴 있는 경우
    직진해도 거리가 줄지 않는 문제가 있었음 — nav_action과 동일한 전략)"""
    dx_body, dy_body, dtheta, pos_err = _pose_error_to(env, state, env.goal_x, env.goal_y, env.goal_heading)
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


def run_demo(render_path=None, max_steps=2000, verbose=True, debug_tail=0):
    env = BusHairpinEnv()
    obs, info = env.reset()

    phase = "NAV"
    wp_idx = 0
    align_step = 0

    best_pos_err = math.inf
    history = []
    terminated = truncated = False
    for t in range(max_steps):
        if phase == "NAV":
            tx, ty = NAV_PATH[wp_idx]
            if wp_idx < len(NAV_PATH) - 1:
                nx, ny = NAV_PATH[wp_idx + 1]
                theading = math.atan2(ny - ty, nx - tx)
            else:
                theading = env.goal_heading
            action, pos_err_to_wp = nav_action(env, env.state, tx, ty, theading)
            if pos_err_to_wp < NAV_CAPTURE_R:
                if wp_idx < len(NAV_PATH) - 1:
                    wp_idx += 1
                else:
                    phase = "ALIGN"
        elif phase == "ALIGN":
            action, dtheta = align_action(env, env.state, env.goal_heading, align_step)
            align_step += 1
            if abs(dtheta) < math.radians(ALIGN_TOL_DEG):
                phase = "FINAL"
        else:  # FINAL
            action, pos_err_final, dtheta_final = final_action(env, env.state)

        obs, reward, terminated, truncated, info = env.step(action)
        _, _, dtheta_g, pos_err = env._pose_error(env.state)
        best_pos_err = min(best_pos_err, pos_err)
        history.append((t, phase, pos_err, math.degrees(dtheta_g), env.state.v, math.degrees(env.state.delta)))
        if terminated or truncated:
            break

    if verbose:
        print(f"steps={t+1} terminated={terminated} truncated={truncated} info={info} best_pos_err={best_pos_err:.2f}")
        if debug_tail:
            for row in history[-debug_tail:]:
                print(f"  t={row[0]} phase={row[1]} pos_err={row[2]:.3f} dtheta={row[3]:.1f}deg v={row[4]:.3f} delta={row[5]:.1f}deg")
    if render_path:
        fig = env.render()
        fig.savefig(render_path, dpi=140)
    return info, best_pos_err, t + 1


if __name__ == "__main__":
    run_demo(render_path="outputs/manual_drive_demo.png", debug_tail=30)
