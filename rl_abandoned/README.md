# rl_abandoned — 폐기된 강화학습(PPO) 갈래

CMA-ES/Hybrid A*로 전환하기 전에 시도했다가 접은 PPO 강화학습 접근의 코드와
학습 산출물입니다. 5단계 경로계획과는 별개의 갈래이며, 결과가 만족스럽지
않아 폐기됐습니다. RL을 다시 볼 계획이 없다면 통째로 삭제해도 무방합니다(약 8MB).

## 구성
- 코드: `train_ppo.py`, `evaluate.py`, RL env 테스트(`test_env.py`, `test_reward.py`)
- 학습 산출물: `bus_hairpin_policy.zip`, `vecnormalize.pkl`, `policy_rollout.png`, `reward_curve.png`
- 로그/백업: `tb_logs/`, `prev_run_backup_20260706/`, `prev_run_backup_20260707_broken/`, `outputs_smoketest_easy/`
- 리포트: `RL_*.md` (빌드/진행/차량기하 리뷰 등)

> 참고: RL 환경 자체(`bus_hairpin_env.py`)와 그것이 import하는 `sensors.py`·
> `reference_path.py`는 경로계획에도 공유되는 토대라 여기가 아니라
> `project_core/envs/`에 있습니다.
