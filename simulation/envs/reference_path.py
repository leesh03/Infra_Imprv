"""중심선 기준 진행률(arc-length progress) 계산 — 보상 shaping용.

도로가 진입 경사로 + 회차 구간 + 진출 경사로로 길어지면서, GOAL까지의
직선거리는 더 이상 실제 주행 가능 경로와 일치하지 않는다(두 경사로가 apex
에서 거의 반대 방향으로 꺾여 있어 직선거리가 줄어드는 방향이 벽을 가로지를
수 있음). 이를 막기 위해 중심선(reference path)에 투영한 진행 거리를
shaping 기준으로 쓴다.
"""
import math
from pathlib import Path

from shapely.geometry import LineString, Point


class ReferencePath:
    def __init__(self, waypoints):
        self.line = LineString(waypoints)
        self.total_length = self.line.length

    @classmethod
    def from_file(cls, path: Path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["waypoints"])

    def arc_length_at(self, x, y):
        return self.line.project(Point(x, y))

    def remaining(self, x, y):
        return self.total_length - self.arc_length_at(x, y)

    def progress_fraction(self, x, y):
        return self.arc_length_at(x, y) / self.total_length

    def lateral_offset(self, x, y):
        return self.line.distance(Point(x, y))

    def point_at(self, s):
        s = max(0.0, min(self.total_length, s))
        p = self.line.interpolate(s)
        return p.x, p.y

    def tangent_heading_at(self, x, y):
        s = self.arc_length_at(x, y)
        eps = 0.5
        x1, y1 = self.point_at(max(0.0, s - eps))
        x2, y2 = self.point_at(min(self.total_length, s + eps))
        return math.atan2(y2 - y1, x2 - x1)
