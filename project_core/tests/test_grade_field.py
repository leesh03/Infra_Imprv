"""TASK 3 완료 기준: 윗/아랫 갈래 임의점이 §2.4 경사값과 일치하고,
정점 전환 구간에서 연속적으로 변하는지(불연속 점프 없음) 검증."""
from geometry.grade_field import GradeField


def test_branch_grades():
    gf = GradeField.from_files()
    assert abs(gf.grade_at(0, 5) - 0.0197) < 1e-3
    assert abs(gf.grade_at(2, -5) - 0.0824) < 1e-3


def test_continuity_across_blend_band():
    gf = GradeField.from_files()
    xs = [x * 0.1 for x in range(-50, 51)]  # x = -5.0 .. 5.0, 정점 부근을 가로지름
    prev = None
    for x in xs:
        g = gf.grade_at(x, 1.0)
        if prev is not None:
            assert abs(g - prev) < 0.01, "정점 전환 구간에서 불연속 점프 발생"
        prev = g
