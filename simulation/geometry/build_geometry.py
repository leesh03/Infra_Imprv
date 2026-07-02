"""TASK 1 실행 스크립트: DXF -> raw_plan.json / grades.json (+ 검증).

사용:
    python -m geometry.build_geometry      (simulation/ 디렉터리에서 실행)
"""
import json
from pathlib import Path

import yaml

from geometry.dxf_parser import (
    extract_bus_wheelbase,
    extract_plan_points,
    extract_slope_profile,
    load_dxf,
)

SIM_DIR = Path(__file__).resolve().parent.parent


def load_config():
    with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    dxf_cfg = cfg["dxf"]
    verify_cfg = cfg["verification"]

    road_path = (SIM_DIR / dxf_cfg["road_dxf"]) if "road_dxf" not in cfg["paths"] else (SIM_DIR / cfg["paths"]["road_dxf"])
    bus_path = SIM_DIR / cfg["paths"]["bus_dxf"]

    road_doc = load_dxf(str(road_path), expected_insunits=dxf_cfg["insunits_mm"])
    bus_doc = load_dxf(str(bus_path), expected_insunits=dxf_cfg["insunits_mm"])

    # --- 평면 점열 ---
    plan_segments = extract_plan_points(
        road_doc, dxf_cfg["plan_layers"], flatten_tol_m=dxf_cfg["flatten_tol_m"]
    )
    geometry_dir = SIM_DIR / cfg["paths"]["geometry_dir"]
    geometry_dir.mkdir(parents=True, exist_ok=True)
    with open(geometry_dir / "raw_plan.json", "w", encoding="utf-8") as f:
        json.dump({"segments": plan_segments}, f, ensure_ascii=False, indent=2)

    # --- 경사 프로파일 ---
    upper = extract_slope_profile(road_doc, dxf_cfg["slope_layers"]["upper"])
    lower = extract_slope_profile(road_doc, dxf_cfg["slope_layers"]["lower"])
    with open(geometry_dir / "grades.json", "w", encoding="utf-8") as f:
        json.dump({"upper": upper, "lower": lower}, f, ensure_ascii=False, indent=2)

    # --- 완료 기준 검증 (§2.4) ---
    tol = verify_cfg["grade_tol_pct"]
    assert abs(upper["grade_pct"] - verify_cfg["upper_grade_pct"]) <= tol, (
        f"윗 경로 경사 불일치: {upper['grade_pct']:.3f}% (기대 {verify_cfg['upper_grade_pct']}% ±{tol})"
    )
    assert abs(lower["grade_pct"] - verify_cfg["lower_grade_pct"]) <= tol, (
        f"아랫 경로 경사 불일치: {lower['grade_pct']:.3f}% (기대 {verify_cfg['lower_grade_pct']}% ±{tol})"
    )

    # --- 버스 축거 재확인 (§2.5) ---
    wheelbase_m = extract_bus_wheelbase(bus_doc)
    wb_tol = verify_cfg["wheelbase_tol_m"]
    expected_wb = cfg["bus"]["wheelbase_L"]
    assert abs(wheelbase_m - expected_wb) <= wb_tol, (
        f"축거 불일치: DXF={wheelbase_m:.4f}m, config={expected_wb}m (허용 ±{wb_tol}m)"
    )

    print(f"평면 세그먼트 수: {len(plan_segments)}")
    print(f"윗 경로  grade={upper['grade_pct']:.3f}%  (run={upper['run_m']:.3f}m, rise={upper['rise_m']:.3f}m)")
    print(f"아랫 경로 grade={lower['grade_pct']:.3f}%  (run={lower['run_m']:.3f}m, rise={lower['rise_m']:.3f}m)")
    print(f"버스 축거 (DXF 재확인) = {wheelbase_m:.4f} m  (config={expected_wb}m)")
    print("TASK 1 완료 기준 통과.")


if __name__ == "__main__":
    main()
