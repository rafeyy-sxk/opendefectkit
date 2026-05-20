"""Shared data contracts for the convert module."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundingBox:
    x_min: float  # absolute pixels
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: Optional[float] = None


@dataclass
class Annotation:
    image_path: str
    image_width: int
    image_height: int
    boxes: List[BoundingBox] = field(default_factory=list)
