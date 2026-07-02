"""DXF -> 평면 점열 / 종단 경사 추출 (TASK 1).

좌표계 규약(RL_BUILD_INSTRUCTIONS.md §1, §2.1 참조):
  - 평면 레이어(Sketch3/Sketch4)는 XY 평면(Z=0) -> 평면 경계 점열.
  - 경사 레이어(윗 경로/아랫 경로)는 XZ 평면(Y=0) -> (run, rise) 프로파일.
  - DXF 단위는 mm. 모든 출력은 m(÷1000).
"""
import math

import ezdxf
import ezdxf.path


def mm_to_m(value):
    return value / 1000.0


def load_dxf(path, expected_insunits=4):
    doc = ezdxf.readfile(path)
    insunits = doc.header.get("$INSUNITS")
    if insunits != expected_insunits:
        raise ValueError(f"{path}: $INSUNITS={insunits}, expected {expected_insunits} (mm)")
    return doc


def extract_plan_points(doc, layer_suffixes, flatten_tol_m=0.02):
    """평면 레이어(suffix로 매칭, 한글 프로젝트 prefix는 무시)의 LINE/SPLINE/ARC/LWPOLYLINE을
    선분 점열로 변환. 각 엔티티 -> [(x,y), ...] (m 단위, Z는 버림).
    """
    msp = doc.modelspace()
    segments = []
    for e in msp:
        layer = e.dxf.layer
        if not any(layer.endswith(suffix) for suffix in layer_suffixes):
            continue
        if e.dxftype() not in ("LINE", "SPLINE", "ARC", "LWPOLYLINE"):
            continue
        path = ezdxf.path.make_path(e)
        pts = [(mm_to_m(p.x), mm_to_m(p.y)) for p in path.flattening(flatten_tol_m * 1000.0)]
        if len(pts) >= 2:
            segments.append({"layer": layer, "type": e.dxftype(), "points": pts})
    return segments


def extract_slope_profile(doc, layer_keyword):
    """경사 프로파일 레이어(XZ 평면, Y=0)의 LINE들에서 (run, rise, grade%) 산출.
    레이어는 직각삼각형 1개로 구성: run = max(x)-min(x), rise = max(z)-min(z) (mm).
    """
    msp = doc.modelspace()
    xs, zs = [], []
    for e in msp:
        if layer_keyword not in e.dxf.layer:
            continue
        if e.dxftype() != "LINE":
            continue
        for pt in (e.dxf.start, e.dxf.end):
            xs.append(pt.x)
            zs.append(pt.z)
    if not xs:
        raise ValueError(f"slope layer containing '{layer_keyword}' not found")
    run_mm = max(xs) - min(xs)
    rise_mm = max(zs) - min(zs)
    run_m = mm_to_m(run_mm)
    rise_m = mm_to_m(rise_mm)
    grade_pct = rise_m / run_m * 100.0
    angle_deg = math.degrees(math.atan(rise_m / run_m))
    return {"run_m": run_m, "rise_m": rise_m, "grade_pct": grade_pct, "angle_deg": angle_deg}


def extract_bus_wheelbase(doc):
    """버스 모델의 바퀴 CIRCLE 중심 좌표 차이로 축거(wheelbase) 재확인. (m)

    DXF 로컬 좌표축 중 분산이 더 큰 축(길이 방향)을 자동 선택한다.
    (모델별로 길이축이 X/Y 어느 쪽으로 들어있을 수 있어 고정 축을 가정하지 않음.)
    """
    msp = doc.modelspace()
    xs, ys = set(), set()
    for e in msp:
        if e.dxftype() == "CIRCLE":
            xs.add(round(e.dxf.center.x, 1))
            ys.add(round(e.dxf.center.y, 1))
    if len(xs) < 2 and len(ys) < 2:
        raise ValueError("bus dxf: 바퀴 CIRCLE 중심을 2개 이상 찾지 못함")
    span_x = max(xs) - min(xs) if len(xs) >= 2 else 0.0
    span_y = max(ys) - min(ys) if len(ys) >= 2 else 0.0
    wheelbase_mm = max(span_x, span_y)
    return mm_to_m(wheelbase_mm)
