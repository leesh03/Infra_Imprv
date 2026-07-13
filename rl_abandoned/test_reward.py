"""TASK 7 완료 기준: 보상 항목별 단위 테스트
(예: 가파른 곳에서 방향전환/정지체류 페널티가 완만한 곳보다 커야 함)."""
import math

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState

# 둘 다 충돌 없음, 여유 충분 (clearance > threshold) 확인된 좌표 (§grade_field 기준 분류)
# heading은 새 GOAL heading(약 pi, 진출 경사로 끝을 향함)에 맞춰 설정 — 도로
# 전체(경사로 포함)로 확장되며 GOAL 위치/heading이 바뀌었기 때문 (TASK1~7 재작업).
UPPER_PT = (6.0, 4.0, math.pi)          # grade ~= 1.97%
LOWER_PT = (28.0, -10.0, math.radians(150))  # grade ~= 8.24%


def make_env():
    return BusHairpinEnv()


def test_pose_shaping_rewards_progress_not_absolute_distance():
    """potential-based shaping: 절대거리가 아니라 '이전 스텝보다 가까워졌는지'에
    보상해야 한다. (절대거리 페널티는 몇 스텝만 방황해도 누적 페널티가 충돌
    페널티를 넘어서 "빨리 충돌해 끝내는" 정책이 더 유리해지는 버그가 있었음 —
    TASK8 1차 학습에서 ep_len_mean=13, 성공률 0%로 수렴해 발견.)"""
    env = make_env()
    far = BicycleState(x=LOWER_PT[0], y=LOWER_PT[1], theta=LOWER_PT[2], v=0.5, delta=0.0)
    near = BicycleState(x=UPPER_PT[0], y=UPPER_PT[1], theta=UPPER_PT[2], v=0.5, delta=0.0)

    r_toward, term_toward, _ = env._reward(far, near)   # 멀리 -> 가까이
    r_away, term_away, _ = env._reward(near, far)        # 가까이 -> 멀리
    assert not term_toward and not term_away
    assert r_toward > 0 > r_away
    assert math.isclose(r_toward + r_away, -2 * env.rcfg["time_penalty"], abs_tol=1e-6)  # 대칭이어야 함

    # 가만히 있으면(거리 변화 없음) pose_term은 0 — 절대거리 페널티가 아님을 보장
    r_still, _, _ = env._reward(far, far)
    assert math.isclose(r_still, -env.rcfg["time_penalty"], abs_tol=1e-6)


def test_reversal_penalty_scales_with_grade():
    """방향전환(전진<->후진) 시 페널티가 가파른 곳에서 더 커야 한다 (§TASK7 표 핵심 항목)."""
    env = make_env()

    def reversal_contribution(x, y, th):
        prev = BicycleState(x=x, y=y, theta=th, v=0.5, delta=0.0)
        same_dir = BicycleState(x=x, y=y, theta=th, v=0.4, delta=0.0)   # 방향전환 없음
        reversed_ = BicycleState(x=x, y=y, theta=th, v=-0.5, delta=0.0)  # 방향전환 있음
        r_same, _, _ = env._reward(prev, same_dir)
        r_rev, _, _ = env._reward(prev, reversed_)
        return r_same - r_rev  # 양수 = 방향전환 페널티 크기

    contrib_shallow = reversal_contribution(*UPPER_PT)
    contrib_steep = reversal_contribution(*LOWER_PT)
    assert contrib_steep > contrib_shallow > 0


def test_stop_on_slope_penalty_scales_with_grade():
    """저속 정지체류 페널티가 가파른 곳에서 더 커야 한다."""
    env = make_env()

    def stop_contribution(x, y, th):
        moving = BicycleState(x=x, y=y, theta=th, v=0.5, delta=0.0)
        idle = BicycleState(x=x, y=y, theta=th, v=0.05, delta=0.0)
        r_moving, _, _ = env._reward(moving, moving)
        r_idle, _, _ = env._reward(idle, idle)
        return r_moving - r_idle  # 양수 = 정지체류 페널티 크기

    contrib_shallow = stop_contribution(*UPPER_PT)
    contrib_steep = stop_contribution(*LOWER_PT)
    assert contrib_steep > contrib_shallow > 0


def test_steer_smoothness_penalizes_large_delta_change():
    env = make_env()
    x, y, th = UPPER_PT
    prev = BicycleState(x=x, y=y, theta=th, v=0.5, delta=0.0)
    small_change = BicycleState(x=x, y=y, theta=th, v=0.5, delta=0.05)
    big_change = BicycleState(x=x, y=y, theta=th, v=0.5, delta=0.3)
    r_small, _, _ = env._reward(prev, small_change)
    r_big, _, _ = env._reward(prev, big_change)
    assert r_big < r_small


def test_clearance_penalty_formula():
    """여유가 임계값보다 작으면 공식대로 페널티가 적용되고, 충분하면 0이어야
    한다. (이 폴리곤/버스 크기 조합에서는 충돌 없이 clearance가 임계값 밑으로
    내려가는 실제 좌표가 없어 — 차체가 크고 여유 폭이 좁아 clearance가
    threshold 밑으로 가기 전에 항상 OBB가 먼저 경계를 침범함 — _clearance를
    임시로 모킹해 공식 자체를 검증한다.)"""
    env = make_env()
    r = env.rcfg
    state = BicycleState(x=UPPER_PT[0], y=UPPER_PT[1], theta=UPPER_PT[2], v=0.5, delta=0.0)
    # prev==state(정지 상태 가정)이므로 potential-based pose_term은 0
    original_clearance = env._clearance

    env._clearance = lambda s: 0.3  # 임계값(1.0m)보다 작은 값으로 모킹
    reward_tight, _, _ = env._reward(state, state)
    expected_clear = -r["w_clear"] * (r["clearance_threshold"] - 0.3)
    assert math.isclose(reward_tight, expected_clear - r["time_penalty"], abs_tol=1e-6)

    env._clearance = lambda s: 5.0  # 임계값보다 충분히 큰 값
    reward_wide, _, _ = env._reward(state, state)
    assert math.isclose(reward_wide, -r["time_penalty"], abs_tol=1e-6)

    env._clearance = original_clearance
    assert reward_tight < reward_wide


def test_goal_bonus_on_success():
    env = make_env()
    theta = env.goal_heading
    rax_x = env.goal_x - env.d_ref * math.cos(theta)
    rax_y = env.goal_y - env.d_ref * math.sin(theta)
    state = BicycleState(x=rax_x, y=rax_y, theta=theta, v=0.0, delta=0.0)
    assert env._is_success(state)

    reward, terminated, info = env._reward(state, state)
    assert terminated and info.get("success")
    assert reward > 0  # goal_bonus가 다른 페널티를 압도해야 함


def test_collision_penalty():
    env = make_env()
    state = BicycleState(x=100.0, y=100.0, theta=0.0, v=0.0, delta=0.0)  # 명백히 경계 밖
    reward, terminated, info = env._reward(state, state)
    assert terminated and info.get("collision")
    assert reward == -env.rcfg["collision_penalty"]
