"""국소 방향전환 제거 — 참조경로(hybrid A*, 403노드, 위반 0으로 이미 검증됨)의
원본 좌표를 재파라미터화하지 않고 그대로 anchor로 쓴다.

evolutionary.py의 전역 CMA-ES 웜스타트는 두 가지 벽에 부딪혔다: (1) 구간이
순차로 이어붙는 사슬 구조라 이른 구간의 작은 변화가 뒤쪽 전체에 누적/증폭돼
지역 정제가 오히려 seed보다 나빠졌고(안전장치로 방지는 했지만 개선은 못
찾음), (2) 원본이 오일러 적분으로 만들어진 경로라 그걸 원호로 통째로
재피팅하면 최대 해상도(K=402)로 쪼개도 ~19m^2의 재구성오차 바닥이 남았다.

이 모듈은 둘 다 피한다: 경로 전체를 하나의 유전자 벡터로 다루지 않고, 28개
방향전환 경계마다 "이 전환 하나를 없앨 수 있는가?"를 독립적으로, 아주 낮은
차원(<=6)의 지역 CMA-ES로 검사한다. 그 전환을 감싸는 두 런(run)의 시작/끝
pose를 원본 데이터에서 그대로 가져와 양 끝을 고정하고, 그 사이를 단일
기어로 잇는 경로를 찾는다 — 성공(위반 0 + 끝 pose 정합)하면 그 구간만 새
연결로 갈아끼우고, 실패하면 원본 그대로 둔다. 그래서 어떤 시도를 해도 절대
원본보다 나빠지지 않는다."""
import math
import multiprocessing
from pathlib import Path

import numpy as np

from planning.evolutionary import arc_pose_at, bus_polygon, decode_genes, sample_segment

MIN_SEG_LEN = 0.05


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def path_json_to_runs(path_json):
    """(start_idx, end_idx, gear) 런 경계 목록 — segments_from_path_json과
    달리 좌표를 원호로 재적합하지 않고 원본 노드 인덱스 경계만 추출한다."""
    rw = path_json["rear_waypoints"]
    gears = path_json["gears"]
    n = len(rw)
    edge_gears = gears[1:n]
    runs = []
    k = 0
    m = len(edge_gears)
    while k < m:
        run_gear = edge_gears[k]
        j = k
        while j < m and edge_gears[j] == run_gear:
            j += 1
        runs.append((k, j, run_gear))
        k = j
    return runs


def _decode_local(gene, k_local, kappa_max, len_cap, gear_sign):
    g = np.clip(np.asarray(gene, dtype=float).reshape(k_local, 2), -1.0, 1.0)
    kappas = g[:, 0] * kappa_max
    ulens = (g[:, 1] * 0.5 + 0.5) * len_cap  # [-1,1] -> [0,len_cap], 방향은 gear_sign 고정
    return list(zip(kappas.tolist(), ulens.tolist()))


def _simulate_local(gene, k_local, entry_pose, gear_sign, kappa_max, len_cap, ds):
    segs = _decode_local(gene, k_local, kappa_max, len_cap, gear_sign)
    x, y, theta = entry_pose
    dense = [(x, y, theta)]
    for kappa, ulen in segs:
        if ulen < 0.02:
            continue
        pts = sample_segment(x, y, theta, kappa, gear_sign * ulen, ds)
        dense.extend(pts)
        x, y, theta = pts[-1]
    return dense, (x, y, theta)


def _local_fitness(gene, k_local, entry_pose, exit_pose, gear_sign, kappa_max, len_cap,
                    footprint, prep_boundary, boundary, ds):
    front_oh, rear_oh, half_width = footprint
    dense, (fx, fy, fth) = _simulate_local(gene, k_local, entry_pose, gear_sign, kappa_max, len_cap, ds)
    violation = 0.0
    for (px, py, pth) in dense[1:]:
        poly = bus_polygon(px, py, pth, front_oh, rear_oh, half_width)
        if not prep_boundary.contains(poly):
            violation += poly.difference(boundary).area
    pos_err = math.hypot(exit_pose[0] - fx, exit_pose[1] - fy)
    head_err = abs(_wrap(exit_pose[2] - fth))
    total_len = sum(u for _, u in _decode_local(gene, k_local, kappa_max, len_cap, gear_sign) if u >= 0.02)
    fit = 500.0 * violation + 20.0 * pos_err + 40.0 * head_err + 0.02 * total_len
    return fit, violation, pos_err, head_err


def try_connect(entry_pose, exit_pose, gear_sign, kappa_max, footprint,
                 fast_boundary, prep_fast, boundary_polygon, prep_precise,
                 k_local=3, len_cap=15.0, ds=0.1, tol_pos=0.08, tol_heading=0.03,
                 maxiter=250, n_restarts=4, seed0=0):
    """entry_pose -> exit_pose를 gear_sign 하나로만 k_local개 이하 원호로 잇는
    시도. 성공하면 dense (x,y,theta) 리스트를, 실패하면 None을 반환한다.
    탐색 루프는 단순화 폴리곤(fast_boundary/prep_fast)으로 빠르게 돌리고,
    후보가 성공 기준을 만족하면 원본 정밀 폴리곤으로 재검증한다 —
    evolutionary.py의 fast/precise 분리와 같은 패턴(원본 16,391정점 폴리곤을
    매 평가/매 try_connect 호출마다 prepare하면 스위치 28개 x 최대 2기어 x
    n_restarts만 해도 감당이 안 됨을 실측으로 확인, maxiter=50/restart=1의
    "스모크 테스트"조차 937초 걸렸다)."""
    import cma

    dim = 2 * k_local

    best = None
    for restart in range(n_restarts):
        seed = seed0 + restart
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-0.4, 0.4, size=dim)
        opts = {"bounds": [-1.0, 1.0], "seed": seed % (2 ** 31 - 1), "maxiter": maxiter, "verbose": -9}
        es = cma.CMAEvolutionStrategy(list(x0), 0.4, opts)
        while not es.stop():
            X = es.ask()
            fits = [
                _local_fitness(g, k_local, entry_pose, exit_pose, gear_sign, kappa_max,
                                len_cap, footprint, prep_fast, fast_boundary, ds)[0]
                for g in X
            ]
            es.tell(X, fits)
        xbest = es.result.xbest
        fit, violation, pos_err, head_err = _local_fitness(
            xbest, k_local, entry_pose, exit_pose, gear_sign, kappa_max, len_cap,
            footprint, prep_fast, fast_boundary, ds,
        )
        fast_ok = violation <= 1e-6 and pos_err <= tol_pos and head_err <= tol_heading
        if best is None or fit < best[0]:
            best = (fit, xbest, fast_ok)
        if fast_ok:
            break

    fit, xbest, fast_ok = best
    if not fast_ok:
        return None
    # 정밀 재검증(원본 폴리곤 + 더 촘촘한 ds)
    fit_p, violation_p, pos_err_p, head_err_p = _local_fitness(
        xbest, k_local, entry_pose, exit_pose, gear_sign, kappa_max, len_cap,
        footprint, prep_precise, boundary_polygon, ds * 0.4,
    )
    if not (violation_p <= 1e-6 and pos_err_p <= tol_pos and head_err_p <= tol_heading):
        return None
    dense, _ = _simulate_local(xbest, k_local, entry_pose, gear_sign, kappa_max, len_cap, ds * 0.4)
    return dense


_WORKER_BOUNDARY = None
_WORKER_FAST_BOUNDARY = None
_WORKER_PREP_PRECISE = None
_WORKER_PREP_FAST = None


def _init_worker(boundary_polygon, fast_boundary):
    # prepared geometry는 pickle이 안 되므로(evolutionary.py와 같은 이유)
    # 프로세스마다 한 번만 로컬로 만든다 — 매 try_connect 호출마다 원본
    # 16,391정점 폴리곤을 prepare()하던 것(스모크 테스트에서 937초 걸림)을
    # 여기서 근본적으로 없앤다.
    global _WORKER_BOUNDARY, _WORKER_FAST_BOUNDARY, _WORKER_PREP_PRECISE, _WORKER_PREP_FAST
    from shapely.prepared import prep
    _WORKER_BOUNDARY = boundary_polygon
    _WORKER_FAST_BOUNDARY = fast_boundary
    _WORKER_PREP_PRECISE = prep(boundary_polygon)
    _WORKER_PREP_FAST = prep(fast_boundary)


def _worker_try_switch(args):
    (i, entry_pose, exit_pose, gear_a, gear_b, kappa_max, footprint,
     k_local, len_cap, maxiter, n_restarts) = args
    for gear_label in (gear_a, gear_b):
        gear_sign = 1.0 if gear_label == "FWD" else -1.0
        dense = try_connect(
            entry_pose, exit_pose, gear_sign, kappa_max, footprint,
            _WORKER_FAST_BOUNDARY, _WORKER_PREP_FAST, _WORKER_BOUNDARY, _WORKER_PREP_PRECISE,
            k_local=k_local, len_cap=len_cap, maxiter=maxiter, n_restarts=n_restarts, seed0=i * 17,
        )
        if dense is not None:
            return i, gear_label, dense
    return i, None, None


def repair_switches(path_json, boundary_polygon, footprint, kappa_max, k_local=3,
                     len_cap=15.0, maxiter=250, n_restarts=4, n_jobs=None, verbose=True):
    """28개(예시) 방향전환 경계 각각을 독립적으로 "제거 가능한가?" 검사한다.
    반환: (runs, results) — results[i]는 switch i(runs[i]<->runs[i+1] 경계)가
    제거 가능하면 (gear_label, dense_path), 불가능하면 None."""
    rw = path_json["rear_waypoints"]
    runs = path_json_to_runs(path_json)
    n_switches = len(runs) - 1

    fast_boundary = boundary_polygon.simplify(0.05, preserve_topology=True).buffer(-0.05)

    tasks = []
    for i in range(n_switches):
        run_a, run_b = runs[i], runs[i + 1]
        entry_pose = tuple(rw[run_a[0]])
        exit_pose = tuple(rw[run_b[1]])
        tasks.append((
            i, entry_pose, exit_pose, run_a[2], run_b[2], kappa_max, footprint,
            k_local, len_cap, maxiter, n_restarts,
        ))

    if n_jobs is None:
        import os
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    results = [None] * n_switches
    if n_jobs > 1:
        with multiprocessing.Pool(n_jobs, initializer=_init_worker,
                                   initargs=(boundary_polygon, fast_boundary)) as pool:
            for i, gear_label, dense in pool.imap_unordered(_worker_try_switch, tasks):
                results[i] = (gear_label, dense) if gear_label else None
                if verbose:
                    run_a, run_b = runs[i], runs[i + 1]
                    status = f"제거 성공(gear={gear_label})" if gear_label else "제거 불가(원본 유지)"
                    print(
                        f"[switch {i + 1}/{n_switches}] "
                        f"run({run_a[2]},{run_a[1] - run_a[0]}nodes) <-> "
                        f"run({run_b[2]},{run_b[1] - run_b[0]}nodes): {status}"
                    )
    else:
        _init_worker(boundary_polygon, fast_boundary)
        for task in tasks:
            i, gear_label, dense = _worker_try_switch(task)
            results[i] = (gear_label, dense) if gear_label else None
            if verbose:
                run_a, run_b = runs[i], runs[i + 1]
                status = f"제거 성공(gear={gear_label})" if gear_label else "제거 불가(원본 유지)"
                print(
                    f"[switch {i + 1}/{n_switches}] "
                    f"run({run_a[2]},{run_a[1] - run_a[0]}nodes) <-> "
                    f"run({run_b[2]},{run_b[1] - run_b[0]}nodes): {status}"
                )
    return runs, results


def splice_repaired_path(path_json, runs, results):
    """겹치지 않는 성공한 제거들을 그리디로 적용해 새 (waypoints,gears,
    rear_waypoints) 경로를 만든다. 실패했거나 이미 다른 제거와 겹치는 구간은
    원본 그대로 둔다."""
    rw = path_json["rear_waypoints"]
    n_runs = len(runs)

    applied = [False] * len(results)
    covered_until = -1
    for i, res in enumerate(results):
        if res is None:
            continue
        run_a, run_b = runs[i], runs[i + 1]
        if run_a[0] <= covered_until:
            continue  # 이전에 적용된 제거와 겹침 -> 건너뜀
        applied[i] = True
        covered_until = run_b[1]

    new_rear = []
    new_gears = []
    i = 0
    node_i = 0
    while node_i < len(rw):
        # node_i가 어떤 적용된 제거의 시작점인지 확인
        matched = None
        for i2, res in enumerate(results):
            if not applied[i2]:
                continue
            run_a, run_b = runs[i2], runs[i2 + 1]
            if run_a[0] == node_i:
                matched = (i2, res, run_a, run_b)
                break
        if matched:
            i2, (gear_label, dense), run_a, run_b = matched
            start_k = len(new_rear)
            if start_k == 0:
                new_rear.append(list(dense[0]))
            for (x, y, th) in dense[1:]:
                new_rear.append([x, y, th])
                new_gears.append(gear_label)
            node_i = run_b[1]
            continue
        if len(new_rear) == 0:
            new_rear.append(list(rw[node_i]))
        else:
            new_rear.append(list(rw[node_i]))
            # 원본 gear 배열은 gears[k]가 (k-1->k) 에지 — node_i에 대응하는 edge gear
            new_gears.append(path_json["gears"][node_i])
        node_i += 1

    n_applied = sum(applied)
    return {"rear_waypoints": new_rear, "gears": ["FWD"] + new_gears}, n_applied


# --------------------------------------------------------------------------- #
# 넓은 창(여러 런을 한 번에), 자유 기어 재최적화 — try_connect(단일 스위치,
# 고정 기어)로는 못 찾는 재구조화를 찾기 위함. 헤어핀 팁처럼 방향전환
# 20개가 1.6m x 2.6m 안에 몰린 조밀한 구간을 통째로 다시 CMA-ES에 맡긴다
# (기어는 evolutionary.py와 같은 부호달린 길이로 자유롭게). 여전히 창 밖은
# 전혀 안 건드리고, 창의 시작/끝 pose(뒷축 x,y,theta)는 원본 데이터 그대로
# 고정한다 — front-center가 아니라 뒷축 pose로 정합을 본다(스플라이스
# 지점에서 이어붙일 다음 원본 노드도 뒷축 좌표이므로).
# --------------------------------------------------------------------------- #


def _simulate_free(gene, K, entry_pose, kappa_max, len_max, ds):
    segs = decode_genes(gene, K, kappa_max, len_max)
    x, y, theta = entry_pose
    dense = [(x, y, theta, "START")]
    n_switch = 0
    prev_gear = None
    for kappa, length in segs:
        if abs(length) < MIN_SEG_LEN:
            continue
        gear = "FWD" if length > 0 else "REV"
        if prev_gear is not None and gear != prev_gear:
            n_switch += 1
        prev_gear = gear
        pts = sample_segment(x, y, theta, kappa, length, ds)
        for (px, py, pth) in pts:
            dense.append((px, py, pth, gear))
        x, y, theta = pts[-1]
    return dense, (x, y, theta), n_switch


def _local_fitness_free(gene, K, entry_pose, exit_pose, kappa_max, len_max, footprint,
                         prep_boundary, boundary, ds, w_switch):
    front_oh, rear_oh, half_width = footprint
    dense, (fx, fy, fth), n_switch = _simulate_free(gene, K, entry_pose, kappa_max, len_max, ds)
    violation = 0.0
    for (px, py, pth, _g) in dense[1:]:
        poly = bus_polygon(px, py, pth, front_oh, rear_oh, half_width)
        if not prep_boundary.contains(poly):
            violation += poly.difference(boundary).area
    pos_err = math.hypot(exit_pose[0] - fx, exit_pose[1] - fy)
    head_err = abs(_wrap(exit_pose[2] - fth))
    total_len = sum(
        abs(l) for _, l in decode_genes(gene, K, kappa_max, len_max) if abs(l) >= MIN_SEG_LEN
    )
    fit = 500.0 * violation + 20.0 * pos_err + 40.0 * head_err + w_switch * n_switch + 0.02 * total_len
    return fit, violation, pos_err, head_err, n_switch


_CW_BOUNDARY = None
_CW_FAST_BOUNDARY = None
_CW_PREP_PRECISE = None
_CW_PREP_FAST = None


def _init_worker_cluster(boundary_polygon, fast_boundary):
    global _CW_BOUNDARY, _CW_FAST_BOUNDARY, _CW_PREP_PRECISE, _CW_PREP_FAST
    from shapely.prepared import prep
    _CW_BOUNDARY = boundary_polygon
    _CW_FAST_BOUNDARY = fast_boundary
    _CW_PREP_PRECISE = prep(boundary_polygon)
    _CW_PREP_FAST = prep(fast_boundary)


def _worker_eval_cluster(args):
    gene, K, entry_pose, exit_pose, kappa_max, len_max, footprint, ds, w_switch = args
    fit, *_ = _local_fitness_free(gene, K, entry_pose, exit_pose, kappa_max, len_max,
                                   footprint, _CW_PREP_FAST, _CW_FAST_BOUNDARY, ds, w_switch)
    return fit


def _worker_random_batch(args):
    # CMA-ES가 이 창에서 350세대x5재시작을 줘도 pos_err~3m에 갇히는데(스위치
    # 수를 세는 항이 만든 불연속 지형 때문으로 추정), 순수 무작위 샘플
    # 20,000개만으로 pos_err=0.10m을 찾은 걸 실측으로 확인했다 — 그래서
    # 넓은 탐색은 CMA 대신 무작위 샘플링에 맡기고, CMA는 그 결과를 정밀
    # 다듬는 좁은 국소 정제(seed-refine과 같은 패턴)에만 쓴다.
    seed, n_samples, K, entry_pose, exit_pose, kappa_max, len_max, footprint, ds, w_switch = args
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_samples):
        gene = rng.uniform(-1.0, 1.0, size=2 * K)
        fit, *_ = _local_fitness_free(gene, K, entry_pose, exit_pose, kappa_max, len_max,
                                       footprint, _CW_PREP_FAST, _CW_FAST_BOUNDARY, ds, w_switch)
        if best is None or fit < best[0]:
            best = (fit, gene)
    return best


def search_cluster(entry_pose, exit_pose, kappa_max, footprint, boundary_polygon,
                    K_list, len_max=8.0, ds=0.1, tol_pos=0.08, tol_heading=0.03,
                    n_random=60000, n_polish_restarts=3, polish_maxiter=200,
                    n_jobs=None, w_switch=3.0, seed0=0, verbose=True):
    """K_list(예: 내림차순으로 [16,12,10,8,6])를 순서대로 시도해, entry_pose->
    exit_pose를 위반 없이 잇는 해 중 가장 스위치 수가 적은 것을 찾는다. 창
    밖은 전혀 손대지 않으므로 원본보다 나빠질 일이 없다(성공 못 하면 None).

    CMA-ES를 처음부터(무작위 x0에서) 이 창에 바로 돌리면 350세대x5재시작을
    줘도 pos_err가 3m대에 갇히는 걸 실측했다(전역 CMA-ES 웜스타트 실험과
    같은 종류의 수렴 실패 — switches를 세는 항이 만든 불연속 지형 때문으로
    추정). 반면 순수 무작위 샘플 20,000개는 pos_err=0.10m을 바로 찾았다.
    그래서 넓은 탐색(reachability)은 병렬 무작위 샘플링에 맡기고, CMA-ES는
    seed-refine과 같은 패턴으로 그 결과 주변 좁은 박스만 다듬는 데 쓴다."""
    import cma

    if n_jobs is None:
        import os
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    fast_boundary = boundary_polygon.simplify(0.05, preserve_topology=True).buffer(-0.05)
    from shapely.prepared import prep
    prep_precise_main = prep(boundary_polygon)

    # 메인 프로세스 자신도 _CW_* 전역을 초기화해야 한다 — pool의 워커들만
    # initializer로 초기화되고, 메인 프로세스에서 직접 부르는 검증 호출
    # (예: random 배치 이후 대표값 재평가)은 별도로 채워두지 않으면
    # _CW_PREP_FAST가 None인 채로 남는다(실측: AttributeError).
    _init_worker_cluster(boundary_polygon, fast_boundary)
    pool = None
    if n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs, initializer=_init_worker_cluster,
                                     initargs=(boundary_polygon, fast_boundary))

    best = None  # (n_switch, dense, K)
    try:
        for K in K_list:
            dim = 2 * K
            # 1단계: 병렬 무작위 샘플링으로 이 K에서 가장 좋은 출발점을 찾는다.
            n_workers = n_jobs if pool is not None else 1
            per_worker = max(1, n_random // n_workers)
            tasks = [
                (seed0 + K * 7919 + w, per_worker, K, entry_pose, exit_pose, kappa_max,
                 len_max, footprint, ds, w_switch)
                for w in range(n_workers)
            ]
            if pool is not None:
                batch_results = pool.map(_worker_random_batch, tasks)
            else:
                batch_results = [_worker_random_batch(t) for t in tasks]
            rand_fit, rand_x0 = min(batch_results, key=lambda r: r[0])
            _, rv, rp, rh, rn = _local_fitness_free(
                rand_x0, K, entry_pose, exit_pose, kappa_max, len_max, footprint,
                _CW_PREP_FAST, _CW_FAST_BOUNDARY, ds, w_switch,
            )
            if verbose:
                print(
                    f"  [K={K} random x{n_random}] fast_violation={rv:.4f} pos_err={rp:.3f} "
                    f"head_err_deg={math.degrees(rh):.2f} n_switch={rn}"
                )

            # 2단계: 그 출발점 주변 좁은 박스에서만 CMA-ES로 정밀 다듬기
            # (seed-refine과 동일 패턴 — 전역 탐색이 아니라 국소 정제).
            k_best = None
            box_eps = 0.05
            box_lower = np.clip(rand_x0 - box_eps, -1.0, 1.0)
            box_upper = np.clip(rand_x0 + box_eps, -1.0, 1.0)
            bounds = [box_lower.tolist(), box_upper.tolist()]
            for restart in range(n_polish_restarts):
                seed = seed0 + K * 1000 + restart
                rng = np.random.default_rng(seed)
                if restart == 0:
                    x0 = rand_x0
                else:
                    noise = rng.normal(0, box_eps * 0.5, size=dim)
                    x0 = np.clip(rand_x0 + noise, box_lower, box_upper)
                opts = {"bounds": bounds, "seed": seed % (2 ** 31 - 1),
                        "maxiter": polish_maxiter, "verbose": -9}
                es = cma.CMAEvolutionStrategy(list(x0), box_eps * 0.5, opts)
                while not es.stop():
                    X = es.ask()
                    if pool is not None:
                        fits = pool.map(_worker_eval_cluster, [
                            (g, K, entry_pose, exit_pose, kappa_max, len_max, footprint, ds, w_switch)
                            for g in X
                        ])
                    else:
                        fits = [
                            _local_fitness_free(g, K, entry_pose, exit_pose, kappa_max, len_max,
                                                 footprint, _CW_PREP_FAST, _CW_FAST_BOUNDARY, ds, w_switch)[0]
                            for g in X
                        ]
                    es.tell(X, fits)
                xbest = es.result.xbest
                fit_p, violation_p, pos_err_p, head_err_p, n_switch_p = _local_fitness_free(
                    xbest, K, entry_pose, exit_pose, kappa_max, len_max, footprint,
                    prep_precise_main, boundary_polygon, ds * 0.5, w_switch,
                )
                success = violation_p <= 1e-6 and pos_err_p <= tol_pos and head_err_p <= tol_heading
                if verbose:
                    print(
                        f"  [K={K} polish restart={restart}] violation={violation_p:.4f} "
                        f"pos_err={pos_err_p:.3f} head_err_deg={math.degrees(head_err_p):.2f} "
                        f"n_switch={n_switch_p} success={success}"
                    )
                if success and (k_best is None or n_switch_p < k_best[0]):
                    dense, _, _ = _simulate_free(xbest, K, entry_pose, kappa_max, len_max, ds * 0.5)
                    k_best = (n_switch_p, dense, K)
            if k_best is not None:
                if best is None or k_best[0] < best[0]:
                    best = k_best
                if verbose:
                    print(f"[K={K}] 이 창에서 최선: n_switch={k_best[0]}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return best  # None 또는 (n_switch, dense, K)
