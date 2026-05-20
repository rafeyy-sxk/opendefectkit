"""Benchmark metrics and evaluation classes for OpenDefectKit."""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ClassMetrics:
    """Per-class detection metrics."""

    label: str
    precision: float
    recall: float
    f1: float
    ap50: float  # AP at IoU=0.5


def _compute_edge_readiness_score(
    model_size_mb: float,
    fps_cpu: Optional[float],
    map50: float,
) -> float:
    """Compute edge readiness score from 0-100 based on size, speed and accuracy."""
    score = 0.0

    # Size component (mutually exclusive)
    if model_size_mb < 10:
        score += 30
    elif model_size_mb < 50:
        score += 20

    # Speed component (mutually exclusive)
    if fps_cpu is not None:
        if fps_cpu > 10:
            score += 30
        elif fps_cpu > 5:
            score += 20

    # Accuracy component (mutually exclusive)
    if map50 > 0.8:
        score += 40
    elif map50 > 0.6:
        score += 25

    return score


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a defect detection model."""

    model_path: str
    dataset_path: str
    map50: float  # mAP@0.5
    map50_95: float  # mAP@0.5:0.95
    per_class: List[ClassMetrics]
    false_positive_rate: float  # FP / (FP + TN) — estimated as 1-precision weighted avg
    false_negative_rate: float  # FN / (FN + TP) — estimated as 1-recall weighted avg
    fps_cpu: Optional[float] = None
    fps_gpu: Optional[float] = None
    model_size_mb: float = 0.0
    edge_readiness_score: float = 0.0  # 0-100

    def print_report(self) -> None:
        """Print a rich-formatted benchmark summary table."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(
                title="Benchmark Report",
                box=box.ROUNDED,
                header_style="bold cyan",
            )

            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("Model Path", self.model_path)
            table.add_row("Dataset Path", self.dataset_path)
            table.add_row("mAP@0.5", f"{self.map50:.4f}")
            table.add_row("mAP@0.5:0.95", f"{self.map50_95:.4f}")
            table.add_row("False Positive Rate", f"{self.false_positive_rate:.4f}")
            table.add_row("False Negative Rate", f"{self.false_negative_rate:.4f}")
            table.add_row("FPS (CPU)", f"{self.fps_cpu:.2f}" if self.fps_cpu is not None else "N/A")
            table.add_row("FPS (GPU)", f"{self.fps_gpu:.2f}" if self.fps_gpu is not None else "N/A")
            table.add_row("Model Size (MB)", f"{self.model_size_mb:.2f}")
            table.add_row("Edge Readiness Score", f"{self.edge_readiness_score:.1f}/100")

            console.print(table)

            if self.per_class:
                class_table = Table(
                    title="Per-Class Metrics",
                    box=box.ROUNDED,
                    header_style="bold magenta",
                )
                class_table.add_column("Class")
                class_table.add_column("Precision", justify="right")
                class_table.add_column("Recall", justify="right")
                class_table.add_column("F1", justify="right")
                class_table.add_column("AP@50", justify="right")

                for cm in self.per_class:
                    f1_style = (
                        "green" if cm.f1 > 0.8 else ("yellow" if cm.f1 > 0.6 else "red")
                    )
                    class_table.add_row(
                        cm.label,
                        f"{cm.precision:.4f}",
                        f"{cm.recall:.4f}",
                        f"[{f1_style}]{cm.f1:.4f}[/{f1_style}]",
                        f"{cm.ap50:.4f}",
                    )

                console.print(class_table)

        except ImportError:
            # Fallback plain text
            print("=== Benchmark Report ===")
            print(f"Model Path:            {self.model_path}")
            print(f"Dataset Path:          {self.dataset_path}")
            print(f"mAP@0.5:               {self.map50:.4f}")
            print(f"mAP@0.5:0.95:          {self.map50_95:.4f}")
            print(f"False Positive Rate:   {self.false_positive_rate:.4f}")
            print(f"False Negative Rate:   {self.false_negative_rate:.4f}")
            fps_cpu_str = f"{self.fps_cpu:.2f}" if self.fps_cpu is not None else "N/A"
            fps_gpu_str = f"{self.fps_gpu:.2f}" if self.fps_gpu is not None else "N/A"
            print(f"FPS (CPU):             {fps_cpu_str}")
            print(f"FPS (GPU):             {fps_gpu_str}")
            print(f"Model Size (MB):       {self.model_size_mb:.2f}")
            print(f"Edge Readiness Score:  {self.edge_readiness_score:.1f}/100")
            if self.per_class:
                print("\n--- Per-Class Metrics ---")
                print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP@50':>10}")
                for cm in self.per_class:
                    print(
                        f"{cm.label:<20} {cm.precision:>10.4f} {cm.recall:>10.4f}"
                        f" {cm.f1:>10.4f} {cm.ap50:>10.4f}"
                    )

    def save_html(self, path: str) -> None:
        """Save a self-contained HTML benchmark report."""
        from opendefectkit.benchmark.report import _render_benchmark_html

        html = _render_benchmark_html(self)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)


class DefectBenchmark:
    """Run a full benchmark of a defect detection model on a test dataset."""

    def __init__(
        self,
        model_path: str,
        test_dataset: str,
        task: str = "defect_detection",
    ) -> None:
        """Initialise benchmark with model path and dataset."""
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.task = task

    def run(self) -> BenchmarkResult:
        """Run validation and return a BenchmarkResult."""
        try:
            import ultralytics
        except ImportError:
            raise ImportError(
                "DefectBenchmark requires ultralytics. "
                "Install with: pip install opendefectkit[benchmark]"
            )

        import numpy as np

        model = ultralytics.YOLO(self.model_path)

        # Run validation
        results = model.val(data=self.test_dataset)

        # Extract mAP metrics
        map50 = float(results.box.map50) if hasattr(results, "box") else 0.0
        map50_95 = float(results.box.map) if hasattr(results, "box") else 0.0

        # Per-class metrics
        per_class: List[ClassMetrics] = []
        if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
            names = model.names
            ap_per_class = results.box.ap50  # shape: (num_classes,)
            p_per_class = results.box.p  # precision per class
            r_per_class = results.box.r  # recall per class
            class_indices = results.box.ap_class_index

            for i, class_idx in enumerate(class_indices):
                label = names.get(int(class_idx), str(class_idx))
                p = float(p_per_class[i]) if i < len(p_per_class) else 0.0
                r = float(r_per_class[i]) if i < len(r_per_class) else 0.0
                ap50_val = float(ap_per_class[i]) if i < len(ap_per_class) else 0.0
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                per_class.append(
                    ClassMetrics(label=label, precision=p, recall=r, f1=f1, ap50=ap50_val)
                )

        # Weighted FPR / FNR estimates
        if per_class:
            avg_precision = sum(cm.precision for cm in per_class) / len(per_class)
            avg_recall = sum(cm.recall for cm in per_class) / len(per_class)
            fpr = 1.0 - avg_precision
            fnr = 1.0 - avg_recall
        else:
            fpr = 0.0
            fnr = 0.0

        # Model size
        model_size_mb = 0.0
        if os.path.isfile(self.model_path):
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        # CPU FPS benchmark — 20 inference passes on a dummy image
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            # Warm-up
            model.predict(dummy, verbose=False)
            start = time.perf_counter()
            for _ in range(20):
                model.predict(dummy, verbose=False)
            elapsed = time.perf_counter() - start
            fps_cpu = 20.0 / elapsed if elapsed > 0 else None
        except Exception:
            fps_cpu = None

        edge_readiness = _compute_edge_readiness_score(model_size_mb, fps_cpu, map50)

        return BenchmarkResult(
            model_path=self.model_path,
            dataset_path=self.test_dataset,
            map50=map50,
            map50_95=map50_95,
            per_class=per_class,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            fps_cpu=fps_cpu,
            fps_gpu=None,
            model_size_mb=model_size_mb,
            edge_readiness_score=edge_readiness,
        )


class ModelComparison:
    """Compare multiple defect detection models side-by-side."""

    def __init__(
        self,
        models: Dict[str, str],  # {name: model_path}
        test_dataset: str,
    ) -> None:
        """Initialise with a dict of model names to paths and a shared dataset."""
        self.models = models
        self.test_dataset = test_dataset
        self._results: Dict[str, BenchmarkResult] = {}

    def run(self) -> Dict[str, BenchmarkResult]:
        """Benchmark each model and return {name: BenchmarkResult}."""
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            raise ImportError(
                "ModelComparison requires ultralytics. "
                "Install with: pip install opendefectkit[benchmark]"
            )

        for name, model_path in self.models.items():
            bench = DefectBenchmark(model_path=model_path, test_dataset=self.test_dataset)
            self._results[name] = bench.run()
        return self._results

    def plot_accuracy_vs_speed(self, save_path: Optional[str] = None) -> None:
        """Scatter plot of mAP50 vs CPU FPS for each model."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, result in self._results.items():
            x = result.fps_cpu if result.fps_cpu is not None else 0.0
            y = result.map50
            ax.scatter(x, y, s=100)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4))

        ax.set_xlabel("FPS (CPU)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title("Accuracy vs Speed")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def generate_report(self, save_path: str) -> None:
        """Generate an HTML comparison report and write to save_path."""
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Embed plot as base64
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, result in self._results.items():
            x = result.fps_cpu if result.fps_cpu is not None else 0.0
            ax.scatter(x, result.map50, s=100)
            ax.annotate(name, (x, result.map50), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("FPS (CPU)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title("Accuracy vs Speed")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Build comparison table rows
        rows_html = ""
        for name, r in self._results.items():
            fps_str = f"{r.fps_cpu:.2f}" if r.fps_cpu is not None else "N/A"
            rows_html += (
                f"<tr><td>{name}</td><td>{r.map50:.4f}</td><td>{r.map50_95:.4f}</td>"
                f"<td>{fps_str}</td><td>{r.model_size_mb:.2f} MB</td>"
                f"<td>{r.edge_readiness_score:.1f}/100</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Comparison Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; background: #f9f9f9; }}
  h1 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  td {{ border: 1px solid #ddd; padding: 8px 12px; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  img {{ max-width: 100%; margin-top: 2em; }}
</style>
</head>
<body>
<h1>Model Comparison Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<table>
  <thead>
    <tr>
      <th>Model</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th>
      <th>FPS (CPU)</th><th>Size</th><th>Edge Score</th>
    </tr>
  </thead>
  <tbody>
{rows_html}  </tbody>
</table>
<h2>Accuracy vs Speed</h2>
<img src="data:image/png;base64,{plot_b64}" alt="Accuracy vs Speed Plot" />
</body>
</html>"""

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(html)
