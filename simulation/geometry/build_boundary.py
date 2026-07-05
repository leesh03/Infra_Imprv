"""TASK 2 실행 스크립트: raw_plan.json -> 닫힌 도로 경계 폴리곤.

사용:
    python -m geometry.build_boundary      (simulation/ 디렉터리에서 실행)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from geometry.polygon_merge import chain_segments

SIM_DIR = Path(__file__).resolve().parent.parent

# v3 dxf: 단일 통합 도면 레이어만 경계 소스로 사용 (Sketch3/Sketch4 구분 없음)
BOUNDARY_LAYER_SUFFIX = "밑 도면"


def main():
    geometry_dir = SIM_DIR / "geometry"
    with open(geometry_dir / "raw_plan.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    boundary_segments = [
        seg for seg in raw["segments"] if seg["layer"].endswith(BOUNDARY_LAYER_SUFFIX)
    ]
    ring, unused = chain_segments(boundary_segments, tol_m=1e-3)
    if unused:
        print(f"[경고] 경계 체인에 들어가지 못한 조각 {len(unused)}개 (보조선으로 추정):")
        for seg in unused:
            print(f"  {seg[0]} -> {seg[-1]}")

    poly = Polygon(ring)
    assert poly.is_valid, "병합된 폴리곤이 유효하지 않음 (self-intersection 등)"
    assert poly.is_simple, "병합된 폴리곤이 단순하지 않음 (자기교차)"
    assert poly.area > 0, "폴리곤 면적이 0 이하"

    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    print(f"점 개수: {len(ring)}, 면적: {poly.area:.2f} m^2")
    print(f"X 범위: [{min(xs):.2f}, {max(xs):.2f}]  Y 범위: [{min(ys):.2f}, {max(ys):.2f}]")

    out = {"polygon": [list(p) for p in ring]}
    with open(geometry_dir / "road_boundary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 장애물/불법주정차 구역 자리 (TASK 2 — 빈 리스트로 마련)
    with open(geometry_dir / "obstacles.json", "w", encoding="utf-8") as f:
        json.dump({"obstacles": []}, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, "b-", linewidth=1.5)
    ax.fill(xs, ys, alpha=0.15, color="blue")
    ax.plot(0, 0, "r+", markersize=12, label="apex (0,0)")
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Road boundary polygon (TASK 2)")
    ax.legend()
    outputs_dir = SIM_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_dir / "road_boundary.png", dpi=150)
    print("저장: geometry/road_boundary.json, outputs/road_boundary.png")
    print("TASK 2 완료 기준 통과.")


if __name__ == "__main__":
    main()
