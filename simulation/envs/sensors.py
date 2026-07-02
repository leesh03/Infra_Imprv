"""경계 여유 센서 — 차량 기준점에서 방사형으로 ray를 쏴 도로 경계/장애물까지의
최근접 거리를 구한다 (TASK 5 observation 항목)."""
import math

from shapely.geometry import LineString


def cast_rays(origin_x, origin_y, heading, n_rays, max_range, boundary_line, obstacle_lines):
    """heading 기준 0~2pi 균등 분포 n_rays개의 거리 리스트(ray가 heading=0일 때
    +X 방향, 차체 좌표계 기준 회전)를 반환. 교차가 없으면 max_range."""
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
                from shapely.geometry import Point as _Point
                pts = [_Point(c) for c in coords]
            for p in pts:
                d = math.hypot(p.x - origin_x, p.y - origin_y)
                if d < best:
                    best = d
        distances.append(best)
    return distances
