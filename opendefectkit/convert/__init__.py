"""Convert module — annotation format conversion for OpenDefectKit."""
from .base import Annotation, BoundingBox
from ._converters import (
    coco_to_yolo,
    yolo_to_coco,
    yolo_to_voc,
    voc_to_yolo,
    labelme_to_yolo,
    auto_detect_and_convert,
    convert_with_label_map,
    detect_format,
)

__all__ = [
    "coco_to_yolo",
    "yolo_to_coco",
    "yolo_to_voc",
    "voc_to_yolo",
    "labelme_to_yolo",
    "auto_detect_and_convert",
    "convert_with_label_map",
    "detect_format",
    "Annotation",
    "BoundingBox",
]
