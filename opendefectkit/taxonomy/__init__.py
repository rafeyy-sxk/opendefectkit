"""Taxonomy module — defect standardization and label mapping for OpenDefectKit."""
import warnings
from typing import Dict, List, Optional

from .defects import ALIASES, DEFECT_REGISTRY, DefectType


class UnknownDefectError(Exception):
    """Raised when a label cannot be matched to any known defect type."""


class DefectTaxonomy:
    """Registry of standard industrial defect types with label normalization utilities."""

    def __init__(self) -> None:
        self._by_name: Dict[str, DefectType] = {d.name: d for d in DEFECT_REGISTRY}
        self._by_name_lower: Dict[str, DefectType] = {
            d.name.lower(): d for d in DEFECT_REGISTRY
        }

    def standardize(self, label: str) -> DefectType:
        """Normalize a raw label string to its canonical DefectType, or raise UnknownDefectError."""
        key = label.strip().lower()

        if key in self._by_name_lower:
            return self._by_name_lower[key]

        if key in ALIASES:
            canonical = ALIASES[key]
            return self._by_name[canonical]

        raise UnknownDefectError(
            f"Label {label!r} does not match any known defect type or alias. "
            f"Use list_defects() to see all registered defects."
        )

    def map_labels(
        self, custom_labels: List[str], method: str = "exact"
    ) -> Dict[str, str]:
        """Map a list of custom labels to standard defect names using the chosen method."""
        if method == "fuzzy_match":
            return self._map_fuzzy(custom_labels)
        return self._map_exact(custom_labels)

    def list_categories(self) -> List[str]:
        """Return the sorted list of all category names in the taxonomy."""
        seen = []
        for defect in DEFECT_REGISTRY:
            if defect.category not in seen:
                seen.append(defect.category)
        return seen

    def list_defects(self, category: Optional[str] = None) -> List[DefectType]:
        """Return all defects, optionally filtered to a single category (case-insensitive)."""
        if category is None:
            return list(DEFECT_REGISTRY)
        cat_lower = category.lower()
        return [d for d in DEFECT_REGISTRY if d.category.lower() == cat_lower]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_exact(self, labels: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for label in labels:
            try:
                defect = self.standardize(label)
                result[label] = defect.name
            except UnknownDefectError:
                result[label] = ""
        return result

    def _map_fuzzy(self, labels: List[str]) -> Dict[str, str]:
        try:
            from rapidfuzz import process as rf_process
            from rapidfuzz import fuzz as rf_fuzz
        except ImportError:
            warnings.warn(
                "rapidfuzz is not installed; falling back to exact matching. "
                "Install it with: pip install opendefectkit[fuzzy]",
                UserWarning,
                stacklevel=3,
            )
            return self._map_exact(labels)

        all_names = list(self._by_name.keys()) + list(ALIASES.keys())
        result: Dict[str, str] = {}
        for label in labels:
            key = label.strip().lower()
            match = rf_process.extractOne(
                key, all_names, scorer=rf_fuzz.WRatio, score_cutoff=70
            )
            if match is None:
                result[label] = ""
            else:
                matched_key = match[0]
                try:
                    defect = self.standardize(matched_key)
                    result[label] = defect.name
                except UnknownDefectError:
                    result[label] = ""
        return result


__all__ = ["DefectType", "DefectTaxonomy", "UnknownDefectError"]
