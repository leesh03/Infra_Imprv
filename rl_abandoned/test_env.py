"""TASK 6 완료 기준(자동화 가능한 부분): gymnasium API 준수, 충돌 정확 감지."""
import numpy as np
from stable_baselines3.common.env_checker import check_env

from envs.bus_hairpin_env import BusHairpinEnv


def test_check_env_sb3():
    env = BusHairpinEnv()
    check_env(env, warn=True)


def test_reset_starts_inside_boundary_without_collision():
    env = BusHairpinEnv()
    env.reset()
    assert not env._check_collision(env.state), "시작 자세 자체가 경계를 침범함"


def test_collision_terminates_episode():
    env = BusHairpinEnv()
    env.reset()
    terminated = False
    for _ in range(200):
        # 최대 가속 + 최대 조향으로 직진 폭주 -> 곧 경계 충돌해야 함
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated and info.get("collision") is True


def test_random_rollout_no_crash_in_step_logic():
    env = BusHairpinEnv()
    obs, info = env.reset(seed=1)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, info = env.reset()
