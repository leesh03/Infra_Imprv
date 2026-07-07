"""중심선 기준 진행률(arc-length progress) 계산 — 보상 shaping용.

도로가 진입 경사로 + 회차 구간 + 진출 경사로로 길어지면서, GOAL까지의
직선거리는 더 이상 실제 주행 가능 경로와 일치하지 않는다(두 경사로가 apex
에서 거의 반대 방향으로 꺾여 있어 직선거리가 줄어드는 방향이 벽을 가로지를
수 있다). 이를 막기 위해 중심선(reference path)에 투영한 진행 거리를
shaping 기준으로 쓴다.
"""
import math
from pathlib import Path

from shapely.geometry import LineString, Point


class ReferencePath:
    def __init__(self, waypoints, rear_waypoints=None):
        self.line = LineString(waypoints)
        self.total_length = self.line.length

        # 2026-07-06: 커리큘럼 시작자세 샘플링용 뒷축 (x,y,theta) 원본.
        # front-center 경로의 접선 방향으로 뒷축 heading을 역산하던 이전
        # 방식은 전방오버행(d_ref)이 R_min과 비슷하거나 커서 급회전 구간에서
        # 크게 틀렸다(직진이 아니면 거의 항상 깨짐) — 그래서 뒷축 데이터를
        # 있는 그대로 저장해두고 여기서는 보간만 한다(역산 없음).
        self._rear_waypoints = rear_waypoints
        if rear_waypoints:
            self._rear_cum = [0.0]
            for i in range(1, len(rear_waypoints)):
                x0, y0, _ = rear_waypoints[i - 1]
                x1, y1, _ = rear_waypoints[i]
                self._rear_cum.append(self._rear_cum[-1] + math.hypot(x1 - x0, y1 - y0))
            self.rear_axle_total_length = self._rear_cum[-1]
        else:
            self._rear_cum = None
            self.rear_axle_total_length = self.total_length

    @classmethod
    def from_file(cls, path: Path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["waypoints"], data.get("rear_waypoints"))

    def arc_length_at(self, x, y):
        return self.line.project(Point(x, y))

    def remaining(self, x, y):
        return self.total_length - self.arc_length_at(x, y)

    def progress_fraction(self, x, y):
        return self.arc_length_at(x, y) / self.total_length

    def lateral_offset(self, x, y):
        return self.line.distance(Point(x, y))

    def rear_axle_pose_at(self, s):
        """뒷축 원본 데이터를 뒷축 기준 arc-length s로 직접 보간한다(역산 없음).
        s는 rear_axle_total_length 기준(=self._rear_cum의 누적거리)."""
        if self._rear_cum is None:
            raise ValueError("rear_waypoints가 없는 ReferencePath에는 쓸 수 없음")
        s = max(0.0, min(self.rear_axle_total_length, s))
        cum = self._rear_cum
        # cum은 단조증가이므로 이분탐색으로 s가 속한 구간 [i-1, i]를 찾는다.
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < s:
                lo = mid + 1
            else:
                hi = mid
        i = max(1, lo)
        s0, s1 = cum[i - 1], cum[i]
        frac = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        x0, y0, th0 = self._rear_waypoints[i - 1]
        x1, y1, th1 = self._rear_waypoints[i]
        x = x0 + (x1 - x0) * frac
        y = y0 + (y1 - y0) * frac
        dtheta = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        theta = th0 + dtheta * frac
        return x, y, theta
