"""Report generation utilities for OpenDefectKit benchmark results."""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendefectkit.benchmark.metrics import BenchmarkResult


def _f1_color(f1: float) -> str:
    """Return an inline CSS background color string based on F1 score."""
    if f1 > 0.8:
        return "#d4edda"  # green
    elif f1 > 0.6:
        return "#fff3cd"  # yellow
    return "#f8d7da"  # red


def _render_benchmark_html(result: "BenchmarkResult") -> str:
    """Render a self-contained HTML benchmark report string."""
    fps_cpu_str = f"{result.fps_cpu:.2f}" if result.fps_cpu is not None else "N/A"
    fps_gpu_str = f"{result.fps_gpu:.2f}" if result.fps_gpu is not None else "N/A"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    per_class_rows = ""
    for cm in result.per_class:
        bg = _f1_color(cm.f1)
        per_class_rows += (
            f'<tr style="background:{bg}">'
            f"<td>{cm.label}</td>"
            f"<td>{cm.precision:.4f}</td>"
            f"<td>{cm.recall:.4f}</td>"
            f"<td>{cm.f1:.4f}</td>"
            f"<td>{cm.ap50:.4f}</td>"
            f"</tr>\n"
        )

    model_filename = result.model_path.split("/")[-1].split("\\")[-1]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Benchmark Report — {model_filename}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; background: #f9f9f9; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }}
  h2 {{ color: #34495e; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em; }}
  th {{
    background: #2c3e50; color: white; padding: 8px 14px;
    text-align: left; font-size: 0.9em;
  }}
  td {{ border: 1px solid #ddd; padding: 8px 14px; font-size: 0.9em; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .metric-key {{ font-weight: bold; width: 220px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-weight: bold; font-size: 0.85em; }}
  .badge-green {{ background: #28a745; color: white; }}
  .badge-yellow {{ background: #ffc107; color: #333; }}
  .badge-red {{ background: #dc3545; color: white; }}
</style>
</head>
<body>
<h1>Benchmark Report</h1>
<p><strong>Generated:</strong> {date_str}</p>

<h2>Model Information</h2>
<table>
  <tr><td class="metric-key">Model Path</td><td>{result.model_path}</td></tr>
  <tr><td class="metric-key">Dataset Path</td><td>{result.dataset_path}</td></tr>
  <tr><td class="metric-key">Model Size</td><td>{result.model_size_mb:.2f} MB</td></tr>
  <tr><td class="metric-key">Edge Readiness Score</td>
      <td>{result.edge_readiness_score:.1f} / 100</td></tr>
</table>

<h2>Detection Metrics</h2>
<table>
  <tr><td class="metric-key">mAP@0.5</td><td>{result.map50:.4f}</td></tr>
  <tr><td class="metric-key">mAP@0.5:0.95</td><td>{result.map50_95:.4f}</td></tr>
  <tr><td class="metric-key">False Positive Rate</td>
      <td>{result.false_positive_rate:.4f}</td></tr>
  <tr><td class="metric-key">False Negative Rate</td>
      <td>{result.false_negative_rate:.4f}</td></tr>
  <tr><td class="metric-key">FPS (CPU)</td><td>{fps_cpu_str}</td></tr>
  <tr><td class="metric-key">FPS (GPU)</td><td>{fps_gpu_str}</td></tr>
</table>

<h2>Per-Class Metrics</h2>
<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>AP@50</th>
    </tr>
  </thead>
  <tbody>
{per_class_rows}  </tbody>
</table>
<p style="font-size:0.8em; color:#666; margin-top:1em;">
  Color legend:
  <span class="badge" style="background:#d4edda">Green</span> F1 &gt; 0.8 &nbsp;
  <span class="badge" style="background:#fff3cd">Yellow</span> F1 &gt; 0.6 &nbsp;
  <span class="badge" style="background:#f8d7da">Red</span> F1 &le; 0.6
</p>
</body>
</html>"""
    return html


class IndustryReportGenerator:
    """Generate professional industry-grade reports from benchmark results."""

    def __init__(
        self,
        model_results: "BenchmarkResult",
        company_name: str,
        inspection_line: str,
    ) -> None:
        """Initialise generator with benchmark results and company metadata."""
        self.model_results = model_results
        self.company_name = company_name
        self.inspection_line = inspection_line

    def _build_recommendation(self) -> str:
        """Return a recommendation string based on FNR, FPR and mAP50."""
        r = self.model_results
        if r.false_negative_rate > 0.2:
            return (
                "HIGH RISK: Model is missing too many defects. Recommend retraining."
            )
        elif r.false_positive_rate > 0.3:
            return (
                "WARNING: High false alarm rate. "
                "Consider confidence threshold tuning."
            )
        elif r.map50 > 0.85:
            return "PRODUCTION READY: Model meets quality threshold."
        else:
            return (
                "REVIEW REQUIRED: Model needs improvement before production deployment."
            )

    def _severity_assessment(self) -> str:
        """Compute an executive severity label."""
        r = self.model_results
        if r.false_negative_rate > 0.2:
            return "HIGH"
        elif r.false_positive_rate > 0.3:
            return "MEDIUM"
        elif r.map50 > 0.85:
            return "LOW"
        return "MEDIUM"

    def generate_pdf(self, output_path: str) -> None:
        """Generate a PDF benchmark report using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
            from reportlab.lib import colors
        except ImportError:
            raise ImportError(
                "generate_pdf requires reportlab. "
                "Install with: pip install opendefectkit[benchmark]"
            )

        r = self.model_results
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        styles = getSampleStyleSheet()

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        elements.append(Paragraph("Defect Detection Benchmark Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Company: {self.company_name}", styles["Normal"]))
        elements.append(Paragraph(f"Inspection Line: {self.inspection_line}", styles["Normal"]))
        elements.append(Paragraph(f"Date: {date_str}", styles["Normal"]))
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Executive Summary", styles["Heading2"]))
        severity = self._severity_assessment()
        elements.append(Paragraph(f"Severity Assessment: {severity}", styles["Normal"]))
        elements.append(Paragraph(self._build_recommendation(), styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Detection Metrics", styles["Heading2"]))
        fps_str = f"{r.fps_cpu:.2f}" if r.fps_cpu is not None else "N/A"
        data = [
            ["Metric", "Value"],
            ["mAP@0.5", f"{r.map50:.4f}"],
            ["mAP@0.5:0.95", f"{r.map50_95:.4f}"],
            ["False Positive Rate", f"{r.false_positive_rate:.4f}"],
            ["False Negative Rate", f"{r.false_negative_rate:.4f}"],
            ["FPS (CPU)", fps_str],
            ["Model Size (MB)", f"{r.model_size_mb:.2f}"],
            ["Edge Readiness Score", f"{r.edge_readiness_score:.1f}/100"],
        ]
        tbl = Table(data, hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(tbl)

        doc.build(elements)

    def generate_html(self, output_path: str) -> None:
        """Generate a professional HTML industry report at output_path."""
        r = self.model_results
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recommendation = self._build_recommendation()
        severity = self._severity_assessment()

        severity_colors = {"HIGH": "#dc3545", "MEDIUM": "#ffc107", "LOW": "#28a745"}
        severity_color = severity_colors.get(severity, "#6c757d")

        fps_cpu_str = f"{r.fps_cpu:.2f}" if r.fps_cpu is not None else "N/A"
        fps_gpu_str = f"{r.fps_gpu:.2f}" if r.fps_gpu is not None else "N/A"

        per_class_rows = ""
        for cm in r.per_class:
            bg = _f1_color(cm.f1)
            per_class_rows += (
                f'<tr style="background:{bg}">'
                f"<td>{cm.label}</td>"
                f"<td>{cm.precision:.4f}</td>"
                f"<td>{cm.recall:.4f}</td>"
                f"<td>{cm.f1:.4f}</td>"
                f"<td>{cm.ap50:.4f}</td>"
                f"</tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Industry Benchmark Report — {self.company_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; color: #333; }}
  .header {{
    background: #2c3e50; color: white; padding: 2em 3em;
  }}
  .header h1 {{ margin: 0 0 0.3em 0; font-size: 1.8em; }}
  .header p {{ margin: 0.2em 0; font-size: 0.95em; opacity: 0.85; }}
  .content {{ padding: 2em 3em; }}
  .severity-badge {{
    display: inline-block; padding: 4px 16px; border-radius: 20px;
    font-weight: bold; font-size: 1em; color: white;
    background: {severity_color};
  }}
  .exec-summary {{
    background: white; border-left: 5px solid {severity_color};
    padding: 1em 1.5em; border-radius: 4px; margin: 1.2em 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}
  .recommendation {{
    background: #eaf4fb; border: 1px solid #bee3f8; padding: 1em 1.5em;
    border-radius: 4px; margin: 1em 0; font-weight: bold;
  }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em;
           background: white; border-radius: 4px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
  th {{
    background: #34495e; color: white; padding: 10px 16px;
    text-align: left; font-size: 0.88em;
  }}
  td {{ border: 1px solid #e0e0e0; padding: 9px 16px; font-size: 0.88em; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .metric-key {{ font-weight: bold; width: 220px; }}
  footer {{ text-align: center; font-size: 0.8em; color: #999; margin: 3em 0 1em; }}
</style>
</head>
<body>
<div class="header">
  <h1>{self.company_name} — Defect Detection Benchmark</h1>
  <p>Inspection Line: {self.inspection_line}</p>
  <p>Report Date: {date_str}</p>
</div>
<div class="content">

  <h2>Executive Summary</h2>
  <div class="exec-summary">
    <p>Severity Assessment: <span class="severity-badge">{severity}</span></p>
  </div>
  <div class="recommendation">{recommendation}</div>

  <h2>Model Information</h2>
  <table>
    <tr><td class="metric-key">Model Path</td><td>{r.model_path}</td></tr>
    <tr><td class="metric-key">Dataset Path</td><td>{r.dataset_path}</td></tr>
    <tr><td class="metric-key">Model Size</td><td>{r.model_size_mb:.2f} MB</td></tr>
    <tr><td class="metric-key">Edge Readiness Score</td>
        <td>{r.edge_readiness_score:.1f} / 100</td></tr>
  </table>

  <h2>Detection Metrics</h2>
  <table>
    <tr><td class="metric-key">mAP@0.5</td><td>{r.map50:.4f}</td></tr>
    <tr><td class="metric-key">mAP@0.5:0.95</td><td>{r.map50_95:.4f}</td></tr>
    <tr><td class="metric-key">False Positive Rate</td>
        <td>{r.false_positive_rate:.4f}</td></tr>
    <tr><td class="metric-key">False Negative Rate</td>
        <td>{r.false_negative_rate:.4f}</td></tr>
    <tr><td class="metric-key">FPS (CPU)</td><td>{fps_cpu_str}</td></tr>
    <tr><td class="metric-key">FPS (GPU)</td><td>{fps_gpu_str}</td></tr>
  </table>

  <h2>Per-Class Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP@50</th>
      </tr>
    </thead>
    <tbody>
{per_class_rows}    </tbody>
  </table>
  <p style="font-size:0.8em;color:#666;margin-top:0.6em;">
    Color legend: green = F1 &gt; 0.8, yellow = F1 &gt; 0.6, red = F1 &le; 0.6
  </p>

  <h2>Recommendations</h2>
  <ul>
    <li>Continuously monitor false negative rate — missed defects are critical in manufacturing.</li>
    <li>Tune confidence thresholds to balance FPR and FNR based on line requirements.</li>
    <li>Retrain periodically with fresh production data to avoid distribution shift.</li>
  </ul>

</div>
<footer>
  Generated by OpenDefectKit &mdash; {date_str}
</footer>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
