"""OpenDefectKit benchmark module — model evaluation and reporting."""

from opendefectkit.benchmark.metrics import (
    BenchmarkResult,
    ClassMetrics,
    DefectBenchmark,
    ModelComparison,
)
from opendefectkit.benchmark.report import IndustryReportGenerator

__all__ = [
    "BenchmarkResult",
    "ClassMetrics",
    "DefectBenchmark",
    "ModelComparison",
    "IndustryReportGenerator",
]
