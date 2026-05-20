"""Standard defect taxonomy data for OpenDefectKit."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DefectType:
    """Canonical representation of a single defect type."""

    id: str
    name: str
    severity_class: str
    category: str


DEFECT_REGISTRY: List[DefectType] = [
    # Surface Cracks
    DefectType("SC-001", "hairline_crack", "medium", "Surface Cracks"),
    DefectType("SC-002", "structural_crack", "critical", "Surface Cracks"),
    DefectType("SC-003", "fatigue_crack", "high", "Surface Cracks"),
    DefectType("SC-004", "thermal_crack", "high", "Surface Cracks"),
    DefectType("SC-005", "stress_crack", "high", "Surface Cracks"),
    # Corrosion
    DefectType("CO-001", "surface_rust", "medium", "Corrosion"),
    DefectType("CO-002", "pitting_corrosion", "high", "Corrosion"),
    DefectType("CO-003", "uniform_corrosion", "medium", "Corrosion"),
    DefectType("CO-004", "galvanic_corrosion", "high", "Corrosion"),
    DefectType("CO-005", "crevice_corrosion", "high", "Corrosion"),
    # Surface Defects
    DefectType("SD-001", "scratch", "low", "Surface Defects"),
    DefectType("SD-002", "gouge", "medium", "Surface Defects"),
    DefectType("SD-003", "dent", "medium", "Surface Defects"),
    DefectType("SD-004", "burr", "low", "Surface Defects"),
    DefectType("SD-005", "chip", "low", "Surface Defects"),
    DefectType("SD-006", "deformation", "high", "Surface Defects"),
    DefectType("SD-007", "inclusion", "medium", "Surface Defects"),
    # Weld Defects
    DefectType("WD-001", "porosity", "medium", "Weld Defects"),
    DefectType("WD-002", "undercut", "high", "Weld Defects"),
    DefectType("WD-003", "incomplete_fusion", "critical", "Weld Defects"),
    DefectType("WD-004", "spatter", "low", "Weld Defects"),
    DefectType("WD-005", "weld_crack", "critical", "Weld Defects"),
    DefectType("WD-006", "overlap", "medium", "Weld Defects"),
    DefectType("WD-007", "burn_through", "high", "Weld Defects"),
    # Coating Defects
    DefectType("CD-001", "blister", "medium", "Coating Defects"),
    DefectType("CD-002", "delamination", "high", "Coating Defects"),
    DefectType("CD-003", "peeling", "medium", "Coating Defects"),
    DefectType("CD-004", "void", "high", "Coating Defects"),
    DefectType("CD-005", "cratering", "low", "Coating Defects"),
    DefectType("CD-006", "fisheye", "low", "Coating Defects"),
    # Dimensional
    DefectType("DI-001", "warping", "high", "Dimensional"),
    DefectType("DI-002", "misalignment", "medium", "Dimensional"),
    DefectType("DI-003", "flash", "low", "Dimensional"),
    DefectType("DI-004", "sink_mark", "medium", "Dimensional"),
    DefectType("DI-005", "short_shot", "critical", "Dimensional"),
]

# Maps alias (lowercase) → canonical defect name
ALIASES: Dict[str, str] = {
    "rust": "surface_rust",
    "corrosion": "surface_rust",
    "oxidation": "surface_rust",
    "fracture": "structural_crack",
    "crack": "structural_crack",
    "cracking": "structural_crack",
    "hairline": "hairline_crack",
    "fatigue": "fatigue_crack",
    "thermal": "thermal_crack",
    "stress": "stress_crack",
    "pitting": "pitting_corrosion",
    "pit": "pitting_corrosion",
    "galvanic": "galvanic_corrosion",
    "crevice": "crevice_corrosion",
    "gouge": "gouge",
    "dent": "dent",
    "mark": "dent",
    "burr": "burr",
    "chip": "chip",
    "deform": "deformation",
    "inclusion": "inclusion",
    "pore": "porosity",
    "porous": "porosity",
    "fusion": "incomplete_fusion",
    "weld_porosity": "porosity",
    "splatter": "spatter",
    "burnthrough": "burn_through",
    "blister": "blister",
    "delaminate": "delamination",
    "peel": "peeling",
    "peeling": "peeling",
    "crater": "cratering",
    "fish_eye": "fisheye",
    "warp": "warping",
    "misalign": "misalignment",
    "flash": "flash",
    "sink": "sink_mark",
    "short": "short_shot",
    "scratch": "scratch",
    "undercut": "undercut",
    "overlap": "overlap",
}
