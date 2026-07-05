"""평면 위 임의 좌표의 종단 경사를 반환하는 경사 맵 (TASK 3).

분할 기준: Sketch3의 두 정점 기준선(윗/아랫 갈래 중심선, 정점에서 출발하는
선분)에 대한 최근접 거리로 갈래를 판정한다. 두 거리가 비슷한 구간(정점
부근)에서는 선형 블렌드로 경사를 보간해 불연속을 없앤다.

각도 이등분선 방식은 정점에서 멀어질수록 부채꼴이 계속 넓어져 실제 도로
형상과 무관하게 먼 영역을 잘못 분류하는 문제가 있었음(예: 우측 끝 영역이
원점 각도만으로 분류됨). 선분까지의 최근접 거리는 정점 부근 분류는 동일하게
유지하면서, 멀리 떨어진 점은 각 선분의 끝점에 거리로 수렴해 영향력이
유한하게 줄어든다는 점에서 더 합리적이다.
"""
import json
import math
from pathlib import Path

import yaml

SIM_DIR = Path(__file__).resolve().parent.parent


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


class GradeField:
    def __init__(self, upper_seg, lower_seg, grade_upper, grade_lower, blend_half_width_m):
        self.upper_seg = upper_seg  # ((x0,y0), (x1,y1))
        self.lower_seg = lower_seg
        self.grade_upper = grade_upper
        self.grade_lower = grade_lower
        self.blend_half_width = blend_half_width_m

    @classmethod
    def from_files(cls, geometry_dir=None, blend_half_width_m=None):
        geometry_dir = Path(geometry_dir) if geometry_dir else (SIM_DIR / "geometry")
        with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if blend_half_width_m is None:
            blend_half_width_m = cfg["slope"]["blend_half_width_m"]
        with open(geometry_dir / "grades.json", "r", encoding="utf-8") as f:
            grades = json.load(f)

        # v3 dxf에는 Sketch3(갈래 판정 기준선)가 없어 config.yaml의 apex 기준
        # 고정 방향 벡터(slope.grade_refs)를 사용한다 (§geometry/build_boundary.py 참조).
        refs = cfg["slope"]["grade_refs"]
        upper_seg = (tuple(refs["upper"][0]), tuple(refs["upper"][1]))
        lower_seg = (tuple(refs["lower"][0]), tuple(refs["lower"][1]))

        return cls(
            upper_seg=upper_seg,
            lower_seg=lower_seg,
            grade_upper=grades["upper"]["grade_pct"] / 100.0,
            grade_lower=grades["lower"]["grade_pct"] / 100.0,
            blend_half_width_m=blend_half_width_m,
        )

    def grade_at(self, x, y):
        d_upper = _point_to_segment_dist(x, y, *self.upper_seg[0], *self.upper_seg[1])
        d_lower = _point_to_segment_dist(x, y, *self.lower_seg[0], *self.lower_seg[1])
        s = d_lower - d_upper  # 양수면 윗 갈래에 더 가까움
        half = self.blend_half_width
        if s > half:
            return self.grade_upper
        if s < -half:
            return self.grade_lower
        t = (s + half) / (2 * half)
        return self.grade_lower + t * (self.grade_upper - self.grade_lower)

    def grade_ahead(self, x, y, theta, d):
        return self.grade_at(x + d * math.cos(theta), y + d * math.sin(theta))


_default_field = None


def grade_at(x, y):
    global _default_field
    if _default_field is None:
        _default_field = GradeField.from_files()
    return _default_field.grade_at(x, y)


def grade_ahead(x, y, theta, d):
    global _default_field
    if _default_field is None:
        _default_field = GradeField.from_files()
    return _default_field.grade_ahead(x, y, theta, d)
