"""
Phase F: Final Robustness Report

Aggregates results from all phases into a comprehensive comparison table
covering the full FP32 / FP16 / INT8 × clean / attacked matrix.

Sources read:
  Classification (DeiT-Tiny, Tiny-ImageNet-200):
    outputs/cls/quant/ptq_results.json   → FP32/FP16/INT8 clean+attacked+ASR+latency

  Detection (RTMDet-Tiny, COCO val2017):
    outputs/det/ptq/det_ptq_results.json → FP32/FP16/INT8 clean+attacked+vanishing_rate+latency

Outputs:
  outputs/reports/robustness_report.md
  outputs/reports/cls_table.png
  outputs/reports/det_table.png

Usage:
    python scripts/final_report.py
    python scripts/final_report.py --out_dir outputs/reports
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.io_utils import ensure_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default result paths
# ---------------------------------------------------------------------------
DEFAULTS = {
    "cls_ptq":   "outputs/cls/quant/ptq_results.json",
    "det_ptq":   "outputs/det/ptq/det_ptq_results.json",
}


def load_json_safe(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Result file not found, skipping: {path}")
        return {}
    with open(p) as f:
        return json.load(f)


def build_cls_rows(ptq: dict) -> list:
    """Build classification table rows from ptq_results.json."""
    rows = []
    for precision in ["FP32", "FP16", "INT8"]:
        m = ptq.get(precision, {})
        if not m:
            continue
        attacked = m.get("attacked_top1_acc")
        asr      = m.get("asr")
        rows.append({
            "Precision":          precision,
            "Clean Top-1 (%)":    f"{m.get('clean_top1_acc', 0)*100:.1f}",
            "Attacked Top-1 (%)": f"{attacked*100:.1f}" if attacked is not None else "N/A",
            "ASR (%)":            f"{asr*100:.1f}" if asr is not None else "N/A",
            "Latency (ms)":       f"{m.get('avg_latency_ms', 0):.2f}",
            "Device":             m.get("device", ""),
        })
    return rows


def build_det_rows(det_ptq: dict) -> list:
    """Build detection table rows from det_ptq_results.json (full precision matrix)."""
    rows = []
    for precision in ["FP32", "FP16", "INT8"]:
        m = det_ptq.get(precision, {})
        if not m:
            continue
        vr  = m.get("vanishing_rate")
        lat = m.get("latency_ms")
        rows.append({
            "Precision":         precision,
            "Clean Boxes/Img":   f"{m.get('clean_avg_boxes', 0):.2f}",
            "Attacked Boxes/Img": f"{m.get('attacked_avg_boxes', 0):.2f}",
            "Vanishing Rate (%)": f"{vr*100:.1f}" if vr is not None else "N/A",
            "Latency (ms)":      f"{lat:.2f}" if lat is not None else "N/A",
            "Device":            m.get("device", ""),
        })
    return rows


def render_md_table(headers: list, rows: list) -> str:
    """Render a markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def plot_table(headers: list, rows: list, title: str, save_path: str):
    """Save table as a PNG using matplotlib."""
    import matplotlib.pyplot as plt

    cell_text = [[str(row.get(h, "")) for h in headers] for row in rows]
    row_labels = [str(i + 1) for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(max(9, len(headers) * 1.7), 1.5 + len(rows) * 0.7))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    plt.title(title, fontsize=13, pad=16)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Table PNG saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase F: Final Robustness Report")
    parser.add_argument("--out_dir", default="outputs/reports")
    parser.add_argument("--cls_ptq", default=DEFAULTS["cls_ptq"])
    parser.add_argument("--det_ptq", default=DEFAULTS["det_ptq"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    print("\n" + "="*65)
    print(" Phase F: Final Robustness Report")
    print("="*65)

    # ------------------------------------------------------------------
    # Load result JSONs
    # ------------------------------------------------------------------
    ptq     = load_json_safe(args.cls_ptq)
    det_ptq = load_json_safe(args.det_ptq)

    # ------------------------------------------------------------------
    # Build table data
    # ------------------------------------------------------------------
    cls_headers = ["Precision", "Clean Top-1 (%)", "Attacked Top-1 (%)",
                   "ASR (%)", "Latency (ms)", "Device"]
    cls_rows = build_cls_rows(ptq)

    det_headers = ["Precision", "Clean Boxes/Img", "Attacked Boxes/Img",
                   "Vanishing Rate (%)", "Latency (ms)", "Device"]
    det_rows = build_det_rows(det_ptq)

    # ------------------------------------------------------------------
    # Render Markdown report
    # ------------------------------------------------------------------
    md_lines = [
        "# Robustness Report: DeiT-Tiny + RTMDet-Tiny",
        "",
        "**Research question:** Does INT8 PTQ change adversarial robustness?",
        "",
        "## Classification (DeiT-Tiny, Tiny-ImageNet-200)",
        "",
        "> Patch: AdvPatch (22×22 px, 1000 steps). Patch optimized on FP32, "
        "transferred to FP16/INT8 without re-optimization.",
        "",
        render_md_table(cls_headers, cls_rows),
        "",
        "## Detection (RTMDet-Tiny, COCO val2017, 200 images)",
        "",
        "> Attack: DPatch (80×80 px, 300 steps). Applied to [0,1] images before "
        "model normalization. Patch optimized on FP32, transferred to FP16/INT8.",
        "",
        render_md_table(det_headers, det_rows),
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate key findings
    if ptq:
        fp32_asr = ptq.get("FP32", {}).get("asr") or 0
        fp16_asr = ptq.get("FP16", {}).get("asr") or 0
        int8_asr = ptq.get("INT8", {}).get("asr") or 0
        md_lines.append(
            f"- **Classification ASR**: FP32={fp32_asr*100:.1f}%, "
            f"FP16={fp16_asr*100:.1f}%, INT8={int8_asr*100:.1f}% "
            f"(Δ INT8-FP32={( int8_asr - fp32_asr)*100:+.1f}%)"
        )
        fp32_lat = ptq.get("FP32", {}).get("avg_latency_ms") or 0
        int8_lat = ptq.get("INT8", {}).get("avg_latency_ms") or 0
        md_lines.append(
            f"- **Cls Latency**: FP32 GPU={fp32_lat:.1f}ms, "
            f"INT8 CPU(ORT)={int8_lat:.1f}ms"
        )

    if det_ptq:
        fp32_vr = det_ptq.get("FP32", {}).get("vanishing_rate") or 0
        fp16_vr = det_ptq.get("FP16", {}).get("vanishing_rate") or 0
        int8_vr = det_ptq.get("INT8", {}).get("vanishing_rate") or 0
        md_lines.append(
            f"- **Detection DPatch vanishing rate**: FP32={fp32_vr*100:.1f}%, "
            f"FP16={fp16_vr*100:.1f}%, INT8={int8_vr*100:.1f}% "
            f"(Δ INT8-FP32={(int8_vr - fp32_vr)*100:+.1f}%)"
        )
        fp32_lat = det_ptq.get("FP32", {}).get("latency_ms") or 0
        fp16_lat = det_ptq.get("FP16", {}).get("latency_ms") or 0
        md_lines.append(
            f"- **Det Latency**: FP32={fp32_lat:.1f}ms, FP16={fp16_lat:.1f}ms per image"
        )

    md_lines += [
        "",
        "## Notes",
        "",
        "- FP16 detection: backbone+neck run in FP16; bbox_head kept FP32 (mmcv.ops.nms requires float32)",
        "- INT8 detection: PyTorch dynamic quantization on backbone Conv2d layers only; "
          "static INT8 for full RTMDet requires mmdeploy",
        "- INT8 classification: ONNX Runtime static quantization (QDQ format) with MinMax calibration",
        "",
        "---",
        "*Generated by scripts/final_report.py*",
        "",
    ]

    report_md = out_dir / "robustness_report.md"
    report_md.write_text("\n".join(md_lines))
    logger.info(f"Markdown report saved: {report_md}")

    # ------------------------------------------------------------------
    # Save PNG tables
    # ------------------------------------------------------------------
    if cls_rows:
        plot_table(cls_headers, cls_rows,
                   "Classification Robustness: DeiT-Tiny (Tiny-ImageNet-200)",
                   str(out_dir / "cls_table.png"))

    if det_rows:
        plot_table(det_headers, det_rows,
                   "Detection Robustness: RTMDet-Tiny (COCO val2017, 200 imgs)",
                   str(out_dir / "det_table.png"))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Classification (DeiT-Tiny / Tiny-ImageNet-200) ===")
    print(f"  {'':6s} {'Clean':>8} {'Attacked':>10} {'ASR':>8} {'Latency':>12}  Device")
    print(f"  {'-'*62}")
    for r in cls_rows:
        print(
            f"  {r['Precision']:<6s} {r['Clean Top-1 (%)']:>7}% "
            f"{r['Attacked Top-1 (%)']:>9}% {r['ASR (%)']:>7}% "
            f"{r['Latency (ms)']:>10}ms  {r['Device']}"
        )

    print("\n=== Detection (RTMDet-Tiny / COCO val2017) ===")
    print(f"  {'':6s} {'Clean Boxes':>12} {'Attacked':>12} {'Vanishing%':>12} {'Latency':>12}  Device")
    print(f"  {'-'*70}")
    for r in det_rows:
        print(
            f"  {r['Precision']:<6s} {r['Clean Boxes/Img']:>12} "
            f"{r['Attacked Boxes/Img']:>12} "
            f"{r['Vanishing Rate (%)']:>12} "
            f"{r['Latency (ms)']:>10}ms  {r['Device']}"
        )

    print(f"\nOutputs:")
    print(f"  Report MD:  {report_md}")
    if cls_rows:
        print(f"  Cls table:  {out_dir}/cls_table.png")
    if det_rows:
        print(f"  Det table:  {out_dir}/det_table.png")
    print()


if __name__ == "__main__":
    main()
