"""경계 여유 센서 — 차량 기준점에서 방사형으로 ray를 쏴 도로 경계/장애물까지의
최근접 거리를 구한다 (TASK 5 observation 항목).

2026-07-06: boundary_line이 16,391개 정점짜리 LineString(flatten_tol_m=0.02로
평탄화된 DXF 곡선 결과 — 대부분 거의 일직선상에 중복 샘플링된 점들)이라, 매
ray마다 이 전체 선분과 intersection을 계산하면 PPO 학습 fps가 51까지
떨어졌다(예상 650의 1/13). STRtree로 개별 선분을 인덱싱해봤지만 후보가
ray 하나당 수천 개로 좁혀지지 않아(경계가 국소적으로도 촘촘함) 파이썬 레벨
intersection 호출 수만 늘어 오히려 더 느려졌다. 대신 ray-casting 전용으로
simplify(tolerance=0.05m)한 사본을 한 번만 만들어 쓴다 — 16,391점이 69점으로
줄고(허용오차 5cm는 20m 레인지 거리센서에는 무의미한 수준), 원래처럼 단순
전체선분 intersection 한 번으로 충분히 빠르다. 충돌판정(_collision_at)은
이 단순화된 선을 쓰지 않고 원본 boundary_polygon.contains()를 그대로 쓰므로
정확도에 영향 없음."""
import math

from shapely.geometry import LineString, Point


def build_sensor_lines(boundary_line, obstacle_lines, tolerance=0.05):
    """ray-casting 전용 단순화 사본을 만든다. env 초기화 시 한 번만 호출."""
    simp_boundary = boundary_line.simplify(tolerance, preserve_topology=True)
    simp_obstacles = [ol.simplify(tolerance, preserve_topology=True) for ol in obstacle_lines]
    return simp_boundary, simp_obstacles


def cast_rays(origin_x, origin_y, heading, n_rays, max_range, boundary_line, obstacle_lines):
    """heading 기준 0~2pi 균등 분포 n_rays개의 거리 리스트(ray가 heading=0일 때
    +X 방향, 차체 좌표계 기준 회전)를 반환. 교차가 없으면 max_range.
    boundary_line/obstacle_lines는 build_sensor_lines()가 만든 단순화 사본이어야
    빠르다(원본 정밀 경계를 넘기면 다시 느려짐)."""
    distances = []
    for i in range(n_rays):
        angle = heading + 2 * math.pi * i / n_rays
        end_x = origin_x + max_range * math.cos(angle)
        end_y = origin_y + max_range * math.sin(angle)
        ray = LineString([(origin_x, origin_y), (end_x, end_y)])

        best = max_range
        for line in (boundary_line, *obstacle_lines):
            inter = ray.intersection(line)
            if inter.is_empty:
                continue
            pts = []
            if inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
            elif inter.geom_type in ("LineString", "MultiLineString"):
                # 접선으로 겹치는 경우 등 — 좌표를 점으로 분해
                coords = []
                geoms = [inter] if inter.geom_type == "LineString" else list(inter.geoms)
                for g in geoms:
                    coords.extend(list(g.coords))
                pts = [Point(c) for c in coords]
            for p in pts:
                d = math.hypot(p.x - origin_x, p.y - origin_y)
                if d < best:
                    best = d
        distances.append(best)
    return distances
