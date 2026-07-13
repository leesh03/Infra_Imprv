# 단계 3 — CMA-ES 처음부터 탐색 (실패)

**결과 이미지:** `cma_path.png` — CMA-ES maneuver: **FAILED (best effort)**, switches=28, len=120.6m, violation=25.6m², goal_pos_err=0.245m

PPO 강화학습 결과가 만족스럽지 않아, 같은 제약(경계·풋프린트·R_min·전진 도착)을
유지한 채 방법만 미분-프리 진화적 최적화(CMA-ES)로 바꿔 회차 기동을 K개 구간의
(방향, 곡률, 길이) 열로 **처음부터** 찾아보려 한 시도입니다. 좁은 통로를 약 70%
지점까지밖에 통과하지 못하고 실패했고, 이 경험에서 "CMA-ES는 좁은 통로 기동을
맨바닥에서 발견하지 못하고, 잘 고정된 국소 리파이너로만 유효하다"는 교훈을 얻어
단계 1(국소 리파이너)로 방향을 틀었습니다.

## 파일
- `evolutionary.py` — 전경로 K-구간 CMA-ES 플래너(`search`, `refine_from_seed`)
- `run_evolutionary_planner.py` — 실행 스크립트
- `cma_path.json` — 실패한 best-effort 산출물(사용 안 함, 참고용)
- `cma_path.png` — 결과 이미지

> 상세 경과는 단계 1 폴더의 `cma_es_investigation_2026-07-08.md`에 함께
> 정리되어 있습니다.
