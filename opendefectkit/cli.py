"""OpenDefectKit CLI — single-file Click interface for all toolkit commands."""
import traceback
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="opendefectkit")
def main() -> None:
    """OpenDefectKit — Industrial computer vision infrastructure."""


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@main.command("convert")
@click.option("--input", "-i", "input_path", required=True, help="Input file or directory.")
@click.option(
    "--from",
    "-f",
    "from_format",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "coco_json", "yolo", "voc_xml", "labelme_json"]),
    help="Source annotation format.",
)
@click.option(
    "--to",
    "-t",
    "to_format",
    required=True,
    type=click.Choice(["yolo", "coco_json", "voc_xml"]),
    help="Target annotation format.",
)
@click.option("--output", "-o", "output_path", required=True, help="Output directory.")
@click.option(
    "--label-map",
    "label_map",
    default=None,
    help="Optional path to a YAML file with label mapping dict.",
)
def convert(
    input_path: str,
    from_format: str,
    to_format: str,
    output_path: str,
    label_map: Optional[str],
) -> None:
    """Convert annotation files between formats."""
    try:
        from opendefectkit.convert import auto_detect_and_convert, convert_with_label_map

        if label_map is not None:
            import yaml

            with open(label_map, "r") as fh:
                mapping = yaml.safe_load(fh)
            convert_with_label_map(input_path, to_format, output_path, mapping)
        else:
            src_fmt = None if from_format == "auto" else from_format
            auto_detect_and_convert(input_path, to_format, output_path, src_fmt)

        console.print(
            Panel(
                f"[bold green]Conversion complete.[/bold green]\n"
                f"  Input  : {input_path}\n"
                f"  Format : {from_format} -> {to_format}\n"
                f"  Output : {output_path}",
                title="[bold cyan]opendefectkit convert[/bold cyan]",
            )
        )
    except ImportError as exc:
        console.print(
            Panel(f"[red]ImportError:[/red] {exc}", title="[bold red]Missing dependency[/bold red]")
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@main.command("analyze")
@click.option("--input", "-i", "input_path", required=True, help="Dataset directory.")
@click.option(
    "--report",
    "-r",
    "report_dir",
    default="reports/",
    show_default=True,
    help="Directory to save reports.",
)
def analyze(input_path: str, report_dir: str) -> None:
    """Profile a dataset and generate an HTML health report."""
    try:
        from opendefectkit.analyze import DatasetHealthScore, DatasetProfiler, DatasetVisualizer

        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        console.print("[bold]Running DatasetProfiler...[/bold]")
        profiler = DatasetProfiler(input_path)
        profile = profiler.run()
        console.print(profile.summary())

        console.print("[bold]Computing DatasetHealthScore...[/bold]")
        health = DatasetHealthScore(input_path).run()
        console.print(str(health))

        console.print("[bold]Generating HTML report...[/bold]")
        visualizer = DatasetVisualizer(input_path)
        html_out = str(report_path / "dataset_report.html")
        visualizer.generate_html_report(html_out)

        console.print(
            Panel(
                f"[bold green]Analysis complete.[/bold green]\n"
                f"  Health score : {health.score}/100\n"
                f"  HTML report  : {html_out}",
                title="[bold cyan]opendefectkit analyze[/bold cyan]",
            )
        )
    except ImportError as exc:
        console.print(
            Panel(f"[red]ImportError:[/red] {exc}", title="[bold red]Missing dependency[/bold red]")
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@main.command("validate")
@click.option("--input", "-i", "input_path", required=True, help="Dataset directory.")
@click.option("--fix", is_flag=True, default=False, help="Run AnnotationFixer after validation.")
@click.option(
    "--output",
    "-o",
    "output_path",
    default=None,
    help="Output directory for fixed dataset (default: {input}_cleaned).",
)
@click.option(
    "--min-box-size",
    "min_box_size",
    default=5,
    show_default=True,
    type=int,
    help="Minimum bounding box dimension in pixels.",
)
def validate(
    input_path: str, fix: bool, output_path: Optional[str], min_box_size: int
) -> None:
    """Validate annotations and optionally fix common issues."""
    try:
        from opendefectkit.validate import AnnotationFixer, AnnotationValidator

        console.print("[bold]Running AnnotationValidator...[/bold]")
        validator = AnnotationValidator(input_path)
        report = validator.check()
        report.print_report()

        console.print(
            Panel(
                f"Total issues found: [bold {'red' if report.has_issues() else 'green'}]"
                f"{len(report.issues)}[/bold {'red' if report.has_issues() else 'green'}]",
                title="[bold cyan]Validation Summary[/bold cyan]",
            )
        )

        if fix:
            out_dir = output_path if output_path else f"{input_path}_cleaned"
            console.print(f"[bold]Running AnnotationFixer — output: {out_dir}[/bold]")
            AnnotationFixer(input_path).fix_all(out_dir, min_box_size=min_box_size)
            console.print(
                Panel(
                    f"[bold green]Fixed dataset saved to:[/bold green] {out_dir}",
                    title="[bold cyan]Fix Complete[/bold cyan]",
                )
            )
    except ImportError as exc:
        console.print(
            Panel(f"[red]ImportError:[/red] {exc}", title="[bold red]Missing dependency[/bold red]")
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------


@main.command("augment")
@click.option("--input", "-i", "input_path", required=True, help="Clean images directory.")
@click.option(
    "--defect",
    "-d",
    "defect_type",
    required=True,
    type=click.Choice(["crack", "rust", "scratch"]),
    help="Type of defect to synthesise.",
)
@click.option(
    "--count",
    "-c",
    "count",
    default=100,
    show_default=True,
    type=int,
    help="Number of synthetic images to generate.",
)
@click.option("--output", "-o", "output_path", required=True, help="Output directory.")
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")
def augment(
    input_path: str, defect_type: str, count: int, output_path: str, seed: int
) -> None:
    """Generate synthetic defect images with paired YOLO annotations."""
    try:
        from opendefectkit.augment import SyntheticDefectGenerator

        generator = SyntheticDefectGenerator(seed=seed)

        console.print(f"[bold]Generating {count} synthetic '{defect_type}' images...[/bold]")

        if defect_type == "crack":
            generator.add_cracks(input_path, output_path, count)
        elif defect_type == "rust":
            generator.add_rust(input_path, output_path, count)
        elif defect_type == "scratch":
            generator.add_scratches(input_path, output_path, count)

        console.print(
            Panel(
                f"[bold green]Augmentation complete.[/bold green]\n"
                f"  Defect type : {defect_type}\n"
                f"  Count       : {count}\n"
                f"  Output      : {output_path}",
                title="[bold cyan]opendefectkit augment[/bold cyan]",
            )
        )
    except ImportError as exc:
        console.print(
            Panel(f"[red]ImportError:[/red] {exc}", title="[bold red]Missing dependency[/bold red]")
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@main.command("benchmark")
@click.option("--model", "-m", "model_path", required=True, help="Model .pt file.")
@click.option("--dataset", "-d", "dataset_path", required=True, help="Test dataset path.")
@click.option(
    "--report",
    "-r",
    "report_dir",
    default="reports/",
    show_default=True,
    help="Report output directory.",
)
def benchmark(model_path: str, dataset_path: str, report_dir: str) -> None:
    """Run a benchmark evaluation of a defect detection model."""
    try:
        from opendefectkit.benchmark import DefectBenchmark

        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        console.print("[bold]Running DefectBenchmark...[/bold]")
        bench = DefectBenchmark(model_path, dataset_path)
        result = bench.run()
        result.print_report()

        html_out = str(report_path / "benchmark_report.html")
        result.save_html(html_out)

        console.print(
            Panel(
                f"[bold green]Benchmark complete.[/bold green]\n"
                f"  mAP@0.5  : {result.map50:.4f}\n"
                f"  HTML     : {html_out}",
                title="[bold cyan]opendefectkit benchmark[/bold cyan]",
            )
        )
    except ImportError as exc:
        console.print(
            Panel(
                f"[red]ImportError:[/red] {exc}\n\n"
                "[dim]Install benchmark deps: "
                "pip install opendefectkit[benchmark][/dim]",
                title="[bold red]Missing dependency[/bold red]",
            )
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------


@main.command("deploy")
@click.option("--model", "-m", "model_path", required=True, help="Model .pt file.")
@click.option(
    "--target",
    "-t",
    "target_device",
    required=True,
    type=click.Choice(
        ["jetson_nano", "jetson_orin", "raspberry_pi_4", "intel_nuc", "generic_x86"]
    ),
    help="Target edge device.",
)
@click.option("--output", "-o", "output_path", required=True, help="Output directory.")
@click.option(
    "--quantize", is_flag=True, default=False, help="Enable INT8 quantization."
)
def deploy(
    model_path: str, target_device: str, output_path: str, quantize: bool
) -> None:
    """Export a model to ONNX and generate an edge deployment package."""
    try:
        from opendefectkit.deploy import EdgeOptimizer, ONNXExporter, DEVICE_PROFILES

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        profile = DEVICE_PROFILES[target_device]
        onnx_path = str(out_dir / "model.onnx")

        console.print(f"[bold]Exporting to ONNX ({target_device})...[/bold]")
        exporter = ONNXExporter(model_path)
        exporter.export(
            onnx_path,
            input_size=profile.recommended_input_size,
            optimize_for="edge",
            quantize=quantize,
        )

        console.print("[bold]Generating deployment package...[/bold]")
        optimizer = EdgeOptimizer(onnx_path)
        optimizer.profile_device(target_device)
        optimizer.generate_deployment_package(str(out_dir))

        console.print(
            Panel(
                f"[bold green]Deployment package ready.[/bold green]\n"
                f"  Target   : {profile.name}\n"
                f"  Quantize : {quantize}\n"
                f"  Output   : {output_path}",
                title="[bold cyan]opendefectkit deploy[/bold cyan]",
            )
        )
    except ImportError as exc:
        console.print(
            Panel(
                f"[red]ImportError:[/red] {exc}\n\n"
                "[dim]Install deploy deps: "
                "pip install opendefectkit[deploy][/dim]",
                title="[bold red]Missing dependency[/bold red]",
            )
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]\n\n[dim]{traceback.format_exc()}[/dim]",
                title="[bold red]Error[/bold red]",
            )
        )
