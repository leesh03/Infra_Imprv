"""평면 경계 조각(LINE/SPLINE 점열)을 하나의 닫힌 외곽 링으로 병합 (TASK 2)."""


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _prune_dead_ends(pool, tol_m):
    """닫힌 링은 모든 꼭짓점의 degree가 2여야 한다. degree-1인 끝점(보조선 등)에
    연결된 조각을 반복적으로 제거해 순수한 폐곡선 조각만 남긴다."""
    pool = list(pool)
    unused = []
    changed = True
    while changed and len(pool) > 1:
        changed = False
        endpoints = [p[0] for p in pool] + [p[-1] for p in pool]
        for i, seg in enumerate(pool):
            for end in (seg[0], seg[-1]):
                degree = sum(1 for ep in endpoints if _dist(ep, end) <= tol_m)
                if degree <= 1:
                    unused.append(pool.pop(i))
                    changed = True
                    break
            if changed:
                break
    return pool, unused


def chain_segments(segments, tol_m=1e-3):
    """끝점-시작점 최근접 매칭으로 segments를 하나의 닫힌 링으로 정렬.

    segments: [{"points": [(x,y), ...]}, ...] (extract_plan_points 출력)
    반환: (ring_points, unused_segments)
        ring_points: 닫힌 링의 점열 (시작점 == 끝점)
        unused_segments: 체인에 들어가지 못한 조각 (경계 외 보조선 등)
    """
    pool = [list(seg["points"]) for seg in segments if len(seg["points"]) >= 2]
    if not pool:
        raise ValueError("no segments to chain")

    pool, pruned = _prune_dead_ends(pool, tol_m)

    chain = pool.pop(0)
    used_any = True
    while used_any and pool:
        used_any = False
        head, tail = chain[0], chain[-1]
        for i, seg in enumerate(pool):
            s_start, s_end = seg[0], seg[-1]
            if _dist(tail, s_start) <= tol_m:
                chain.extend(seg[1:])
                pool.pop(i)
                used_any = True
                break
            if _dist(tail, s_end) <= tol_m:
                chain.extend(list(reversed(seg))[1:])
                pool.pop(i)
                used_any = True
                break
            if _dist(head, s_end) <= tol_m:
                chain[:0] = seg[:-1]
                pool.pop(i)
                used_any = True
                break
            if _dist(head, s_start) <= tol_m:
                chain[:0] = list(reversed(seg))[:-1]
                pool.pop(i)
                used_any = True
                break
        if _dist(chain[0], chain[-1]) <= tol_m:
            break  # 루프 닫힘

    if _dist(chain[0], chain[-1]) > tol_m:
        raise ValueError(
            f"체인이 닫히지 않음: 시작 {chain[0]} vs 끝 {chain[-1]} (거리={_dist(chain[0], chain[-1]):.4f}m). "
            f"미사용 조각 {len(pool)}개로도 닫히지 않음 - 끊긴 구간 존재 가능성."
        )

    return chain, pruned + pool
