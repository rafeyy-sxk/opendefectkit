"""Dataset visualization utilities for OpenDefectKit."""
import base64
import io
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opendefectkit.analyze.profiler import DatasetProfiler, DatasetHealthScore


class DatasetVisualizer:
    """Generates charts and HTML reports for a YOLO-format dataset."""

    def __init__(self, dataset_dir: str) -> None:
        """Initialise with the root directory of the dataset."""
        self.dataset_dir = dataset_dir
        self._profiler = DatasetProfiler(dataset_dir)
        self._report = self._profiler.run()
        self._health = DatasetHealthScore(dataset_dir).run()

    def plot_class_distribution(self, save_path: str) -> None:
        """Save a bar chart of class distribution to save_path (PNG)."""
        report = self._report
        classes = list(report.class_distribution.keys())
        counts = list(report.class_distribution.values())

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 5))
        ax.bar(classes, counts, color="#2196F3", edgecolor="#1565C0")
        ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)

    def plot_defect_heatmap(self, save_path: str) -> None:
        """Save a 10x10 defect density heatmap to save_path (PNG)."""
        grid = np.array(self._report.defect_density_grid)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="hot", vmin=0, vmax=max(grid.max(), 1e-6))
        plt.colorbar(im, ax=ax, label="Normalised density")
        ax.set_title("Defect Density Heatmap (10×10 grid)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        plt.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)

    def plot_bbox_sizes(self, save_path: str) -> None:
        """Save a scatter plot of bbox widths vs heights to save_path (PNG)."""
        widths = self._report.bbox_widths
        heights = self._report.bbox_heights
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(widths, heights, alpha=0.5, s=10, color="#E91E63")
        ax.set_title("Bounding Box Sizes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        plt.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)

    # ── HTML report ──────────────────────────────────────────────────────────

    def _fig_to_base64(self, fig: "plt.Figure") -> str:
        """Convert a matplotlib figure to a base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def generate_html_report(self, save_path: str) -> None:
        """Generate a self-contained HTML report with embedded charts and save to save_path."""
        report = self._report
        health = self._health

        # ── Build plots in memory ────────────────────────────────────────────
        # Class distribution
        classes = list(report.class_distribution.keys())
        counts = list(report.class_distribution.values())
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.bar(classes, counts, color="#2196F3", edgecolor="#1565C0")
        ax1.set_title("Class Distribution")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        b64_dist = self._fig_to_base64(fig1)
        plt.close(fig1)

        # Defect heatmap
        grid = np.array(report.defect_density_grid)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(grid, cmap="hot", vmin=0, vmax=max(grid.max(), 1e-6))
        plt.colorbar(im, ax=ax2)
        ax2.set_title("Defect Density Heatmap")
        plt.tight_layout()
        b64_heatmap = self._fig_to_base64(fig2)
        plt.close(fig2)

        # BBox scatter
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(report.bbox_widths, report.bbox_heights, alpha=0.5, s=10, color="#E91E63")
        ax3.set_title("Bounding Box Sizes")
        ax3.set_xlabel("Width (px)")
        ax3.set_ylabel("Height (px)")
        plt.tight_layout()
        b64_bbox = self._fig_to_base64(fig3)
        plt.close(fig3)

        # ── Class table rows ─────────────────────────────────────────────────
        class_rows = "".join(
            f"<tr><td>{cls}</td><td>{cnt}</td></tr>"
            for cls, cnt in report.class_distribution.items()
        )

        # ── Health check rows ─────────────────────────────────────────────────
        def _badge(passed: Optional[bool]) -> str:
            if passed is True:
                return '<span style="color:green;font-weight:bold">PASS</span>'
            if passed is False:
                return '<span style="color:red;font-weight:bold">FAIL</span>'
            return '<span style="color:orange;font-weight:bold">WARN</span>'

        health_rows = "".join(
            f"<tr><td>{item.criterion}</td><td>{_badge(item.passed)}</td>"
            f"<td>{item.detail}</td><td>{item.weight}</td></tr>"
            for item in health.items
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenDefectKit — Dataset Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
  h1 {{ color: #1565C0; }}
  h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin-bottom: 24px; width: 100%; max-width: 700px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #e3f2fd; }}
  .metric {{ display: inline-block; margin: 8px 24px 8px 0; }}
  .metric span {{ font-size: 2em; font-weight: bold; color: #1565C0; }}
  .score {{ font-size: 3em; font-weight: bold; color: {'#2e7d32' if health.score >= 70 else '#f57f17' if health.score >= 40 else '#c62828'}; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
  .plots {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .plots figure {{ margin: 0; }}
</style>
</head>
<body>
<h1>OpenDefectKit — Dataset Profile Report</h1>

<h2>Overview</h2>
<div>
  <div class="metric">Total images <br><span>{report.total_images}</span></div>
  <div class="metric">Annotated <br><span>{report.annotated_images}</span></div>
  <div class="metric">Unannotated <br><span>{report.unannotated_images}</span></div>
  <div class="metric">Duplicates <br><span>{report.duplicate_count}</span></div>
  <div class="metric">Low-quality <br><span>{len(report.low_quality_images)}</span></div>
</div>

<h2>Health Score</h2>
<div class="score">{health.score} / 100</div>
<p><strong>Recommendation:</strong> {health.recommendation}</p>
<table>
  <tr><th>Criterion</th><th>Status</th><th>Detail</th><th>Weight</th></tr>
  {health_rows}
</table>

<h2>Class Distribution</h2>
<table>
  <tr><th>Class</th><th>Count</th></tr>
  {class_rows}
</table>
<figure>
  <img src="data:image/png;base64,{b64_dist}" alt="Class distribution chart">
</figure>

<h2>Defect Density Heatmap</h2>
<figure>
  <img src="data:image/png;base64,{b64_heatmap}" alt="Defect density heatmap">
</figure>

<h2>Bounding Box Size Distribution</h2>
<figure>
  <img src="data:image/png;base64,{b64_bbox}" alt="Bounding box sizes">
</figure>

<h2>Recommended Split</h2>
<table>
  <tr><th>Split</th><th>Images</th></tr>
  <tr><td>Train</td><td>{report.recommended_split.get('train', 0)}</td></tr>
  <tr><td>Val</td><td>{report.recommended_split.get('val', 0)}</td></tr>
  <tr><td>Test</td><td>{report.recommended_split.get('test', 0)}</td></tr>
</table>

<footer style="margin-top:32px;color:#999;font-size:0.85em">
  Generated by OpenDefectKit
</footer>
</body>
</html>"""

        Path(save_path).write_text(html, encoding="utf-8")
