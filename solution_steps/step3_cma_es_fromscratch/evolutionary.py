"""K개 (곡률, 부호달린 길이) 구간의 열로 기동을 표현하고 CMA-ES로 직접
최적화하는 미분-프리 플래너 — PPO 강화학습을 대체한다.

사용자가 제안한 표현은 구간당 (방향 dᵢ∈{FWD,REV}, 곡률 κᵢ, 길이 Lᵢ)의
3-튜플이었는데, 여기서는 부호달린 길이 sᵢ(부호=방향, 절댓값=길이) 하나로
합쳐 구간당 (κᵢ, sᵢ) 2-튜플로 쓴다 — 표현력은 동일하면서 CMA-ES가 다루는
연속 파라미터가 3K개가 아니라 2K개가 되고, 방향을 위한 별도의 이산 변수를
연속 최적화기에 맞게 임계값-판정할 필요도 없어진다.

각 구간은 뒷축 기준 상수-곡률 원호이며 닫힌형 공식으로 정확히 적분한다
(오일러 누적이 아님 — 후진은 같은 원을 반대로 되짚는 것과 동일, kinematics
의 dtheta=κ*ds 관계에서 그대로 따라나옴).

fitness = 경계/장애물 침범 면적 + 목표 자세 오차 + 방향전환 페널티(경사
가중, config.yaml reward.w_reversal/reversal_slope_k 재사용 — RL과 동일한
제약/조건을 그대로 씀) + 총 길이 + 전진도착 위반 페널티.

경계 검사는 sensors.py가 겪은 것과 같은 문제(원본 경계 폴리곤이 16,391개
정점)를 겪는다 — CMA-ES는 fitness를 수만 번 평가하므로 매번 정밀 폴리곤에
대고 검사하면 감당이 안 된다. 그래서 단순화+안쪽 버퍼(안전마진)를 준 폴리곤
으로 탐색하고, 성공 후보만 원본 정밀 폴리곤 + 더 촘촘한 표본간격으로
재검증한다(sensors.py의 build_sensor_lines와 같은 해법 패턴).
"""
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.prepared import prep

from kinematics.bicycle import front_center

# 사용자 하드 요구사항(memory: project_new_dxf_path_replan — "0 trailing REV
# steps, 앞면이 닿는 전진 도착"): 마지막으로 실제 움직인 구간은 반드시 FWD여야
# 하고, 최소 이만큼은 연속으로 전진해야 "진짜 전진 도착"으로 인정한다(하이브리드
# A*의 arrival_min_fwd_run=3 grid-step 개념을 물리적 거리로 옮긴 것).
FWD_ARRIVAL_MIN_LEN = 1.0
# 이보다 짧은 구간은 "미사용"으로 보고 시뮬레이션에서 건너뛴다 — CMA-ES가
# K를 다 채우지 않고 일부 구간을 0으로 눌러 실질적으로 더 적은 구간 수를
# 쓰는 것을 허용한다(성장하는 K 루프와 별개로 자연스러운 자유도).
MIN_SEG_LEN = 0.05


def arc_pose_at(x0, y0, theta0, kappa, l):
    """상수 곡률 kappa, 부호달린 원호길이 l(양수=전진, 음수=후진) 이동 후의
    (x,y,theta) — 닫힌형이므로 임의의 l에 대해 오차 없이 정확하다."""
    theta1 = theta0 + kappa * l
    if abs(kappa) < 1e-9:
        return x0 + l * math.cos(theta0), y0 + l * math.sin(theta0), theta1
    R = 1.0 / kappa
    x1 = x0 + R * (math.sin(theta1) - math.sin(theta0))
    y1 = y0 - R * (math.cos(theta1) - math.cos(theta0))
    return x1, y1, theta1


def sample_segment(x0, y0, theta0, kappa, length, ds):
    """|length|를 ds 근방으로 등분해 정확히 세그먼트 끝에서 끝나는 표본열을
    반환한다(끝점 포함, 시작점 제외 — 이전 구간의 끝점과 중복되지 않게)."""
    n = max(1, math.ceil(abs(length) / ds))
    step = length / n
    pts = []
    x, y, theta = x0, y0, theta0
    for _ in range(n):
        x, y, theta = arc_pose_at(x, y, theta, kappa, step)
        pts.append((x, y, theta))
    return pts


def decode_genes(gene_vec, K, kappa_max, len_max):
    g = np.clip(np.asarray(gene_vec, dtype=float).reshape(K, 2), -1.0, 1.0)
    kappas = g[:, 0] * kappa_max
    lengths = g[:, 1] * len_max
    return list(zip(kappas.tolist(), lengths.tolist()))


def encode_segments(segments, kappa_max, len_max):
    """decode_genes의 역변환 — (kappa,length) 리스트를 [-1,1]^(2K) 유전자로
    인코딩한다(CMA-ES 웜스타트 초기값 만들 때 사용)."""
    g = []
    for kappa, length in segments:
        g.append(max(-1.0, min(1.0, kappa / kappa_max)) if kappa_max else 0.0)
        g.append(max(-1.0, min(1.0, length / len_max)) if len_max else 0.0)
    return np.asarray(g, dtype=float)


def segments_from_path_json(path_json, chunk_len=2.0):
    """run_hybrid_astar.py가 저장한 reference_path.json 포맷(rear_waypoints
    + gears)을 (kappa, 부호달린 길이) 세그먼트 목록으로 근사 변환한다.
    CMA-ES 웜스타트 초기값 용도라 완벽한 재현이 목적은 아니지만, 최소한
    "재구성했을 때 충돌 없이 목표에 도착"할 정도의 정확도는 있어야 웜스타트로
    쓸모가 있다.

    처음에는 같은 gear가 이어지는 구간 전체(런)를 상수-곡률 원호 하나로
    근사했는데, 최장 런(39m)처럼 런 내부에서 실제로 곡률이 상당히 바뀌는
    경우 "런 전체 평균곡률" 원호가 진짜 모양과 크게 어긋나 재구성 위치오차가
    19m까지 벌어지는 게 실측으로 확인됐다(끝 헤딩은 맞아도 중간 경로가
    완전히 다른 모양이 됨). 그래서 런을 그대로 쓰지 않고 런 내부를 다시
    chunk_len 근방으로 잘게 쪼개, 각 조각마다 국소 평균곡률(그 조각 안의
    인접 노드 간 헤딩차 누적 / 조각 길이)을 쓴다 — 조각이 충분히 짧으면
    (곡률반경 R_min=7.37m 기준 2m 조각은 최대 ~15.6도 이내) 상수곡률 근사
    오차가 무시할 만큼 작다. gear 방향전환 횟수(switches)는 런 경계에서만
    세므로 조각을 잘게 쪼개도 늘어나지 않는다.

    gears[i](i=1..n-1)는 rear_waypoints[i-1]->[i] 에지의 기어다(gears[0]은
    루트 노드용 더미값 — hybrid_astar.py의 _reconstruct/run_hybrid_astar.py
    저장 로직과 대응). 조각 내 헤딩차는 인접 노드 간(각 에지 ~0.3m로 아주
    작음 — 모호성 없이 wrap 가능) 값을 그대로 누적해 winding을 보존한다
    (끝-시작 theta 차이를 통째로 wrap하면 큰 회전에서 틀어짐)."""
    rw = path_json["rear_waypoints"]
    gears = path_json["gears"]
    n = len(rw)
    if n < 2:
        return []
    edge_gears = gears[1:n]
    edge_len = [math.hypot(rw[i + 1][0] - rw[i][0], rw[i + 1][1] - rw[i][1]) for i in range(n - 1)]
    edge_dtheta = [
        (rw[i + 1][2] - rw[i][2] + math.pi) % (2 * math.pi) - math.pi
        for i in range(n - 1)
    ]
    segments = []
    k = 0
    m = len(edge_gears)
    while k < m:
        run_gear = edge_gears[k]
        j = k
        while j < m and edge_gears[j] == run_gear:
            j += 1
        sign = 1.0 if run_gear == "FWD" else -1.0
        acc_len = 0.0
        acc_dtheta = 0.0
        for e in range(k, j):
            acc_len += edge_len[e]
            acc_dtheta += edge_dtheta[e]
            if acc_len >= chunk_len or e == j - 1:
                # arc_pose_at은 dtheta = kappa * (부호달린 길이)로 적분한다.
                # acc_dtheta는 실제 관측된(이미 올바른 부호의) 헤딩변화량이고
                # 부호달린 길이는 sign*acc_len이므로, kappa = acc_dtheta /
                # (sign*acc_len) = sign*acc_dtheta/acc_len (sign=±1이라
                # 1/sign=sign). sign을 빼먹으면 REV 구간마다 헤딩변화가
                # 반대로 재구성된다(실측으로 확인한 버그).
                kappa = sign * acc_dtheta / acc_len if acc_len > 1e-6 else 0.0
                segments.append((kappa, sign * acc_len))
                acc_len = 0.0
                acc_dtheta = 0.0
        k = j
    return segments


def bus_polygon(x, y, theta, front_oh, rear_oh, half_width):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front_oh, half_width), (front_oh, -half_width),
              (-rear_oh, -half_width), (-rear_oh, half_width)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])


@dataclass
class PlannerContext:
    start_pose: Tuple[float, float, float]
    goal_x: float
    goal_y: float
    goal_heading: float
    tol_pos: float
    tol_heading: float
    footprint: Tuple[float, float, float]  # (front_oh, rear_oh, half_width)
    d_ref: float
    L: float
    kappa_max: float
    len_max: float
    boundary_polygon: Polygon
    obstacle_polygons: list
    grade_field: object
    reference_path: object = None
    w_violation: float = 500.0
    w_goal_pos: float = 20.0
    w_goal_heading: float = 20.0
    w_reversal: float = 2.0
    reversal_slope_k: float = 5.0
    w_length: float = 0.05
    w_arrival: float = 80.0
    ds: float = 0.25          # 탐색(fitness 수만 회 평가)용 — 빠르게
    verify_ds: float = 0.05   # 승자 최종 재검증용 — 촘촘하게
    fast_boundary: Optional[Polygon] = field(default=None, repr=False)
    _prep_fast: object = field(default=None, repr=False)
    _prep_precise: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.fast_boundary is None:
            simplified = self.boundary_polygon.simplify(0.05, preserve_topology=True)
            # 안쪽으로 살짝 줄여 단순화 오차를 보수적으로 흡수(sensors.py의
            # 같은 문제에 대한 해법 패턴 — simplify(0.05)는 "형상 변화 없음"
            # 수준이라 알려져 있지만, fitness 탐색 단계에서는 그래도 안전
            # 마진을 둔다. 최종 승자는 이 폴리곤이 아니라 원본으로 재검증).
            self.fast_boundary = simplified.buffer(-0.05)
        self._prep_fast = prep(self.fast_boundary)
        self._prep_precise = prep(self.boundary_polygon)

    def boundary_for(self, precise):
        return self.boundary_polygon if precise else self.fast_boundary

    def prep_for(self, precise):
        return self._prep_precise if precise else self._prep_fast


def evaluate(gene_vec, K, ctx: PlannerContext, precise=False):
    """gene_vec(길이 2K, [-1,1] 범위)을 디코드해 기동을 시뮬레이션하고
    fitness(작을수록 좋음)와 진단 정보를 반환한다.
    precise=True면 원본 정밀 경계 폴리곤 + 촘촘한 표본간격으로 재검증한다
    (탐색 중에는 항상 False — 속도용 단순화 폴리곤 사용)."""
    boundary = ctx.boundary_for(precise)
    prep_boundary = ctx.prep_for(precise)
    ds = ctx.verify_ds if precise else ctx.ds

    front_oh, rear_oh, half_width = ctx.footprint
    x, y, theta = ctx.start_pose
    segments = decode_genes(gene_vec, K, ctx.kappa_max, ctx.len_max)

    total_violation = 0.0
    total_length = 0.0
    n_switches = 0
    weighted_switches = 0.0
    prev_gear = None
    last_active_gear = None
    last_active_len = 0.0
    dense = [(x, y, theta, "START")]

    for kappa, length in segments:
        if abs(length) < MIN_SEG_LEN:
            continue
        gear = "FWD" if length > 0 else "REV"
        if prev_gear is not None and gear != prev_gear:
            grade = ctx.grade_field.grade_at(x, y)
            weighted_switches += 1.0 + ctx.reversal_slope_k * grade
            n_switches += 1
        prev_gear = gear
        last_active_gear = gear
        last_active_len = abs(length)
        total_length += abs(length)

        pts = sample_segment(x, y, theta, kappa, length, ds)
        for (px, py, pth) in pts:
            poly = bus_polygon(px, py, pth, front_oh, rear_oh, half_width)
            if not prep_boundary.contains(poly):
                total_violation += poly.difference(boundary).area
            for obs in ctx.obstacle_polygons:
                if poly.intersects(obs):
                    total_violation += poly.intersection(obs).area
            dense.append((px, py, pth, gear))
        x, y, theta = pts[-1]

    fc_x, fc_y = front_center(x, y, theta, ctx.d_ref)
    goal_pos_err = math.hypot(ctx.goal_x - fc_x, ctx.goal_y - fc_y)
    dtheta = (ctx.goal_heading - theta + math.pi) % (2 * math.pi) - math.pi
    goal_heading_err = abs(dtheta)

    # reference_path.py의 경고와 동일한 문제: 진입/진출 경사로가 굽어 있어
    # 목표까지의 직선거리가 줄어드는 방향이 벽을 가로지를 수 있다 — 초기
    # 실험에서 실제로 CMA-ES가 이 함정에 갇혀 K를 늘려도 goal_pos_err가
    # 전혀 줄지 않는 현상을 확인했다(모든 K에서 18~20m 근방에 고정). RL
    # 보상(_progress_metric)이 쓴 것과 같은 해법 — 이미 검증된 기준선
    # reference_path에 투영한 "남은 경로 길이"를 fitness의 위치 항으로
    # 쓴다(성공 판정 자체는 여전히 아래 goal_pos_err 직선거리 기준 그대로).
    pos_progress = ctx.reference_path.remaining(fc_x, fc_y) if ctx.reference_path else goal_pos_err

    arrival_ok = (last_active_gear == "FWD" and last_active_len >= FWD_ARRIVAL_MIN_LEN)
    arrival_penalty = 0.0 if arrival_ok else ctx.w_arrival

    fitness = (
        ctx.w_violation * total_violation
        + ctx.w_goal_pos * pos_progress
        + ctx.w_goal_heading * ctx.L * goal_heading_err
        + ctx.w_reversal * weighted_switches
        + ctx.w_length * total_length
        + arrival_penalty
    )

    success = (
        total_violation <= 1e-6
        and goal_pos_err <= ctx.tol_pos
        and goal_heading_err <= ctx.tol_heading
        and arrival_ok
    )

    meta = {
        "fitness": fitness, "violation": total_violation, "goal_pos_err": goal_pos_err,
        "pos_progress": pos_progress,
        "goal_heading_err": goal_heading_err, "n_switches": n_switches,
        "weighted_switches": weighted_switches, "total_length": total_length,
        "arrival_ok": arrival_ok, "success": success,
        "final_pose": (x, y, theta), "dense_path": dense, "segments": segments,
    }
    return fitness, meta


# --------------------------------------------------------------------------- #
# 병렬 fitness 평가용 워커. Windows(spawn)에서는 프로세스 간에 전달되는 값이
# 전부 pickle되어야 하는데, shapely의 prepared geometry(PlannerContext 내부
# _prep_fast/_prep_precise)는 pickle이 안 되는 C-익스텐션 객체다. 그래서
# ctx를 매 호출마다 넘기지 않고, 워커 프로세스마다 한 번만(Pool 생성 시)
# "prepare 없는" 생성자 kwargs를 넘겨 각자 로컬로 PlannerContext를 다시
# 만들게 한다(__post_init__이 그 프로세스 안에서 자기 prep 객체를 새로 만듦).
# ---------------------------------------------------------------------------- #
_WORKER_CTX = None


def _init_worker(ctx_kwargs):
    global _WORKER_CTX
    _WORKER_CTX = PlannerContext(**ctx_kwargs)


def _worker_eval(args):
    gene_vec, K = args
    fit, _ = evaluate(gene_vec, K, _WORKER_CTX, precise=False)
    return fit


def _make_pool(ctx_kwargs, n_jobs):
    import multiprocessing
    if n_jobs <= 1:
        return None
    return multiprocessing.Pool(n_jobs, initializer=_init_worker, initargs=(ctx_kwargs,))


def run_cma(K, ctx, ctx_kwargs, x0, sigma0, maxiter, popsize, pool, seed, bounds=None):
    import cma
    opts = {"bounds": bounds if bounds is not None else [-1.0, 1.0],
            "seed": seed % (2 ** 31 - 1), "maxiter": maxiter, "verbose": -9}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        X = es.ask()
        if pool is not None:
            fits = pool.map(_worker_eval, [(x, K) for x in X])
        else:
            fits = [evaluate(x, K, ctx, precise=False)[0] for x in X]
        es.tell(X, fits)
    return np.asarray(es.result.xbest), es.result.fbest


def search(ctx_kwargs, K_start=1, K_max=12, n_restarts=3, maxiter=200, popsize=None,
           sigma0=0.5, seed0=0, n_jobs=None, verbose=True):
    """K=K_start부터 하나씩 늘려가며(warm-start) CMA-ES를 돌리다가, 경계
    침범 없이 목표(위치/헤딩 허용오차) + 전진도착 조건을 모두 만족하는 첫
    해를 찾으면 즉시 반환한다. K_max까지도 못 찾으면 시도 중 fitness가
    가장 좋았던 해를 success=False로 반환한다(빈손으로 끝나지 않게).

    반환: (K_used, xbest, meta, success) — meta는 항상 정밀 폴리곤 기준
    재평가 결과(final_pose/dense_path 등을 신뢰해도 됨)."""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    ctx = PlannerContext(**ctx_kwargs)
    pool = _make_pool(ctx_kwargs, n_jobs)

    best_overall_fit = math.inf
    best_overall = None  # (K, xbest)
    warm_x0 = None

    try:
        for K in range(K_start, K_max + 1):
            dim = 2 * K
            for restart in range(n_restarts):
                seed = seed0 + K * 1000 + restart
                rng = np.random.default_rng(seed)
                pad_n = dim - len(warm_x0) if warm_x0 is not None else dim
                if restart == 0 and warm_x0 is not None and pad_n >= 0:
                    x0 = np.concatenate([warm_x0, rng.uniform(-0.2, 0.2, size=pad_n)])
                else:
                    x0 = rng.uniform(-0.3, 0.3, size=dim)

                xbest, _ = run_cma(K, ctx, ctx_kwargs, x0, sigma0, maxiter, popsize, pool, seed)
                fit, meta = evaluate(xbest, K, ctx, precise=False)

                if verbose:
                    print(
                        f"[K={K} restart={restart}] fit={fit:.3f} violation={meta['violation']:.4f} "
                        f"pos_progress(remaining)={meta['pos_progress']:.2f}m "
                        f"goal_pos_err={meta['goal_pos_err']:.3f}m "
                        f"goal_heading_err={math.degrees(meta['goal_heading_err']):.2f}deg "
                        f"switches={meta['n_switches']} len={meta['total_length']:.1f}m "
                        f"arrival_ok={meta['arrival_ok']}"
                    )

                if fit < best_overall_fit:
                    best_overall_fit = fit
                    best_overall = (K, xbest)
                    warm_x0 = xbest

                if meta["success"]:
                    fit_p, meta_p = evaluate(xbest, K, ctx, precise=True)
                    if meta_p["success"]:
                        if verbose:
                            print(f"[K={K}] SUCCESS — 원본 정밀 폴리곤 재검증 통과")
                        return K, xbest, meta_p, True
                    if verbose:
                        print(
                            f"[K={K}] 단순화 폴리곤은 통과했지만 정밀 재검증 실패 "
                            f"(violation={meta_p['violation']:.4f}m^2) — 계속 탐색"
                        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    K_b, xbest_b = best_overall
    _, meta_p = evaluate(xbest_b, K_b, ctx, precise=True)
    return K_b, xbest_b, meta_p, False


def refine_from_seed(ctx_kwargs, seed_segments, n_restarts=4, maxiter=300, sigma0=0.03,
                      box_eps=0.08, seed0=0, n_jobs=None, verbose=True):
    """이미 충돌 없이 목표에 도달하는 것으로 알려진 seed_segments(예:
    hybrid A* 참조경로를 segments_from_path_json으로 근사 변환한 것)를
    CMA-ES 출발점으로 삼아 같은 K(=len(seed_segments))에서 지역적으로
    정제한다. 미로 전체를 처음부터 다시 찾는 게 아니라 "이미 뚫려 있는
    경로를 부드럽게 다듬고 불필요한 방향전환을 줄이는" 역할 — 순수
    무작위 시작 CMA-ES가 이 도로의 좁은 헤어핀 포켓을 뚫지 못하고 경로의
    ~70% 지점에서 반복적으로 막히는 것을 실측으로 확인한 뒤 채택.

    구간이 순차적으로 이어 붙는 구조라(한 구간의 시작점이 이전 구간 전체의
    결과) 이른 구간의 아주 작은 변화도 뒤쪽 구간 전체에 누적/증폭된다 —
    실측으로 확인: sigma0=0.03·[-1,1] 전역 bounds인 채로 CMA-ES를 50세대만
    돌려도 위반면적이 seed의 27m^2에서 3300m^2로 폭주했다(28회였던 방향전환도
    40회로 늘어남). 그래서 매 좌표마다 seed±box_eps로 하드 클립된 bounds를
    줘서, CMA의 step-size 적응이 무엇을 하든 seed 근방 박스를 물리적으로
    벗어날 수 없게 만든다(전역 sigma0/작은 초기값만으로는 이 사슬형 민감도
    구조 때문에 안 막힘 — 실측으로 확인).

    MIN_SEG_LEN 미만으로 줄어드는 구간은 switch/length 페널티에 의해
    자연스럽게 병합·제거되므로, 방향전환 횟수는 seed의 것(예: 28회)보다
    줄어들 수 있다(늘어나진 않도록 성공 후보 중 fitness 최솟값만 채택)."""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)
    K = len(seed_segments)
    ctx = PlannerContext(**ctx_kwargs)
    pool = _make_pool(ctx_kwargs, n_jobs)
    seed_x0 = encode_segments(seed_segments, ctx.kappa_max, ctx.len_max)
    box_lower = np.clip(seed_x0 - box_eps, -1.0, 1.0)
    box_upper = np.clip(seed_x0 + box_eps, -1.0, 1.0)
    # 길이 유전자(홀수 인덱스)는 부호(=기어)를 절대 못 넘도록 0에서 자른다
    # — box_eps만으로는 짧은 구간(예: 0.5m)이 넉넉한 절대폭(box_eps*len_max)
    # 안에서 쉽게 부호가 뒤집혀 새 방향전환을 만들어낸다(실측: 28->71회).
    # 0을 못 건너가게 하드 클립하면 어떤 세대에서도 switches가 seed보다
    # 늘어날 수 없다(구간이 0으로 줄어 사실상 사라지는 감소 방향은 여전히 허용).
    len_gene_idx = np.arange(1, 2 * K, 2)
    pos_mask = seed_x0[len_gene_idx] > 0
    box_lower[len_gene_idx[pos_mask]] = np.maximum(box_lower[len_gene_idx[pos_mask]], 0.0)
    neg_mask = seed_x0[len_gene_idx] < 0
    box_upper[len_gene_idx[neg_mask]] = np.minimum(box_upper[len_gene_idx[neg_mask]], 0.0)
    bounds = [box_lower.tolist(), box_upper.tolist()]

    # CMA-ES는 x0 자체를 후보로 평가하지 않는다(1세대는 N(x0,sigma0^2*C)에서
    # 뽑은 표본들이고 x0는 그 평균일 뿐) — seed가 아주 좁은 가능영역(벽에
    # 바짝 붙은 헤어핀 구간)에 있으면 224차원 동시 섭동이 거의 항상 어딘가
    # 하나는 벽을 침범해, "이번 배치에서 그나마 나은 것"이 seed 자체보다도
    # 나쁠 수 있다(실측: violation 27.7m^2 -> 233.7m^2로 후퇴). 그래서
    # seed의 fitness를 기준선으로 깔아두고, 어떤 재시작도 이보다 못하면
    # seed 자체를 최종 결과로 되돌린다(정제가 절대 원본보다 나빠지지 않게).
    if verbose:
        print(f"[seed-refine] box_eps={box_eps} sigma0={sigma0} K={K} (dim={2*K})")
    seed_fit, seed_meta = evaluate(seed_x0, K, ctx, precise=False)
    if verbose:
        print(
            f"[seed-refine baseline] fit={seed_fit:.3f} violation={seed_meta['violation']:.4f} "
            f"goal_pos_err={seed_meta['goal_pos_err']:.3f}m switches={seed_meta['n_switches']} "
            f"len={seed_meta['total_length']:.1f}m"
        )
    # 채택 기준은 반드시 정밀(원본 폴리곤) 평가여야 한다 — fast-eval(단순화+
    # 안쪽버퍼 폴리곤) 기준으로만 seed와 비교했더니, 단순화 폴리곤에서는
    # seed보다 좋아 보이지만(위반↓) 정밀 재검증하면 오히려 더 나쁜(위반↑,
    # switches 28->71) 후보가 "개선"으로 잘못 채택되는 걸 실측으로 확인했다.
    # restart 수는 적지만(정밀 평가 1회는 fast-eval 수만 번에 비해 무시할
    # 만큼 저렴) 매 restart 끝에 정밀 재평가로 진짜 비교한다.
    seed_fit_p, seed_meta_p = evaluate(seed_x0, K, ctx, precise=True)
    best_fit_p = seed_fit_p
    best_xbest = seed_x0
    best_meta_p = seed_meta_p
    if verbose:
        print(
            f"[seed-refine baseline precise] fit={seed_fit_p:.3f} violation={seed_meta_p['violation']:.4f} "
            f"goal_pos_err={seed_meta_p['goal_pos_err']:.3f}m switches={seed_meta_p['n_switches']}"
        )
    try:
        for restart in range(n_restarts):
            seed = seed0 + restart
            rng = np.random.default_rng(seed)
            if restart == 0:
                x0 = seed_x0
            else:
                noise = rng.normal(0, box_eps * 0.5, size=2 * K)
                x0 = np.clip(seed_x0 + noise, box_lower, box_upper)

            xbest, _ = run_cma(K, ctx, ctx_kwargs, x0, sigma0, maxiter, None, pool, seed, bounds=bounds)
            fit, meta = evaluate(xbest, K, ctx, precise=False)
            fit_p, meta_p = evaluate(xbest, K, ctx, precise=True)

            if verbose:
                print(
                    f"[seed-refine restart={restart}] fast_fit={fit:.3f} fast_violation={meta['violation']:.4f} | "
                    f"precise_fit={fit_p:.3f} precise_violation={meta_p['violation']:.4f} "
                    f"goal_pos_err={meta_p['goal_pos_err']:.3f}m "
                    f"goal_heading_err={math.degrees(meta_p['goal_heading_err']):.2f}deg "
                    f"switches={meta_p['n_switches']} len={meta_p['total_length']:.1f}m "
                    f"arrival_ok={meta_p['arrival_ok']}"
                )

            if fit_p < best_fit_p:
                best_fit_p, best_xbest, best_meta_p = fit_p, xbest, meta_p

            if meta_p["success"]:
                if verbose:
                    print("[seed-refine] SUCCESS — 정밀 폴리곤 기준 성공")
                return K, xbest, meta_p, True
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return K, best_xbest, best_meta_p, False
