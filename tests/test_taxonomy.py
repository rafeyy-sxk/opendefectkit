"""Tests for the opendefectkit.taxonomy module."""
import pytest

from opendefectkit.taxonomy import DefectTaxonomy, DefectType, UnknownDefectError


@pytest.fixture
def taxonomy() -> DefectTaxonomy:
    return DefectTaxonomy()


def test_standardize_exact_name(taxonomy: DefectTaxonomy) -> None:
    defect = taxonomy.standardize("surface_rust")
    assert isinstance(defect, DefectType)
    assert defect.name == "surface_rust"


def test_standardize_alias(taxonomy: DefectTaxonomy) -> None:
    defect = taxonomy.standardize("rust")
    assert defect.name == "surface_rust"


def test_standardize_case_insensitive(taxonomy: DefectTaxonomy) -> None:
    defect = taxonomy.standardize("CRACK")
    assert defect.name == "structural_crack"


def test_standardize_unknown_raises(taxonomy: DefectTaxonomy) -> None:
    with pytest.raises(UnknownDefectError):
        taxonomy.standardize("blorp")


def test_map_labels_exact(taxonomy: DefectTaxonomy) -> None:
    result = taxonomy.map_labels(["rust", "scratch"], method="exact")
    assert result["rust"] == "surface_rust"
    assert result["scratch"] == "scratch"


def test_list_categories(taxonomy: DefectTaxonomy) -> None:
    categories = taxonomy.list_categories()
    assert len(categories) == 6
    assert "Surface Cracks" in categories
    assert "Corrosion" in categories
    assert "Surface Defects" in categories
    assert "Weld Defects" in categories
    assert "Coating Defects" in categories
    assert "Dimensional" in categories


def test_list_defects_filtered(taxonomy: DefectTaxonomy) -> None:
    corrosion_defects = taxonomy.list_defects("Corrosion")
    assert len(corrosion_defects) == 5
    assert all(d.category == "Corrosion" for d in corrosion_defects)
    names = [d.name for d in corrosion_defects]
    assert "surface_rust" in names
    assert "pitting_corrosion" in names


def test_all_defects_count(taxonomy: DefectTaxonomy) -> None:
    all_defects = taxonomy.list_defects()
    assert len(all_defects) >= 30


def test_defect_type_fields(taxonomy: DefectTaxonomy) -> None:
    defect = taxonomy.standardize("porosity")
    assert defect.id
    assert defect.name == "porosity"
    assert defect.severity_class in ("low", "medium", "high", "critical")
    assert defect.category == "Weld Defects"


def test_severity_classes_valid(taxonomy: DefectTaxonomy) -> None:
    valid = {"low", "medium", "high", "critical"}
    for defect in taxonomy.list_defects():
        assert defect.severity_class in valid, (
            f"{defect.name} has invalid severity_class={defect.severity_class!r}"
        )


def test_map_labels_unknown_returns_empty_string(taxonomy: DefectTaxonomy) -> None:
    result = taxonomy.map_labels(["notadefect"], method="exact")
    assert result["notadefect"] == ""


def test_list_defects_case_insensitive_filter(taxonomy: DefectTaxonomy) -> None:
    lower = taxonomy.list_defects("corrosion")
    upper = taxonomy.list_defects("Corrosion")
    assert lower == upper


def test_map_labels_fuzzy_fallback(taxonomy: DefectTaxonomy) -> None:
    # Without rapidfuzz installed the method should still return a dict
    result = taxonomy.map_labels(["rust", "scratch"], method="fuzzy_match")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"rust", "scratch"}
