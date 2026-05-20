"""Tests for opendefectkit.augment module."""
from __future__ import annotations


import cv2
import numpy as np
import pytest

from opendefectkit.augment import IndustrialAugPipeline, SyntheticDefectGenerator


# ---------------------------------------------------------------------------
# SyntheticDefectGenerator — cracks
# ---------------------------------------------------------------------------

class TestAddCracks:
    def test_add_cracks_creates_output_images(self, sample_images_dir, tmp_dir):
        """num_samples=3 produces exactly 3 images in output_dir/images/."""
        out = tmp_dir / "crack_out"
        gen = SyntheticDefectGenerator(seed=42)
        gen.add_cracks(str(sample_images_dir), str(out), num_samples=3)
        images = list((out / "images").glob("*.jpg"))
        assert len(images) == 3

    def test_add_cracks_creates_yolo_labels(self, sample_images_dir, tmp_dir):
        """3 corresponding .txt label files are created."""
        out = tmp_dir / "crack_out"
        gen = SyntheticDefectGenerator(seed=42)
        gen.add_cracks(str(sample_images_dir), str(out), num_samples=3)
        labels = list((out / "labels").glob("*.txt"))
        assert len(labels) == 3

    def test_add_cracks_classes_txt(self, sample_images_dir, tmp_dir):
        """classes.txt exists and contains 'crack'."""
        out = tmp_dir / "crack_out"
        gen = SyntheticDefectGenerator(seed=42)
        gen.add_cracks(str(sample_images_dir), str(out), num_samples=2)
        classes_file = out / "classes.txt"
        assert classes_file.exists()
        assert "crack" in classes_file.read_text()

    def test_add_cracks_label_format(self, sample_images_dir, tmp_dir):
        """Each label file has 5 space-separated floats, all in [0, 1]."""
        out = tmp_dir / "crack_out"
        gen = SyntheticDefectGenerator(seed=42)
        gen.add_cracks(str(sample_images_dir), str(out), num_samples=3)
        for lbl in (out / "labels").glob("*.txt"):
            parts = lbl.read_text().strip().split()
            assert len(parts) == 5, f"Expected 5 values, got {len(parts)} in {lbl}"
            vals = [float(p) for p in parts]
            assert all(0.0 <= v <= 1.0 for v in vals), f"Values out of [0,1]: {vals}"


# ---------------------------------------------------------------------------
# SyntheticDefectGenerator — rust & scratches
# ---------------------------------------------------------------------------

class TestAddRust:
    def test_add_rust_creates_output(self, sample_images_dir, tmp_dir):
        """num_samples=2 produces 2 images."""
        out = tmp_dir / "rust_out"
        gen = SyntheticDefectGenerator(seed=7)
        gen.add_rust(str(sample_images_dir), str(out), num_samples=2)
        images = list((out / "images").glob("*.jpg"))
        assert len(images) == 2


class TestAddScratches:
    def test_add_scratches_creates_output(self, sample_images_dir, tmp_dir):
        """num_samples=2 produces 2 images."""
        out = tmp_dir / "scratch_out"
        gen = SyntheticDefectGenerator(seed=99)
        gen.add_scratches(str(sample_images_dir), str(out), num_samples=2)
        images = list((out / "images").glob("*.jpg"))
        assert len(images) == 2


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_seeded_reproducibility(self, sample_images_dir, tmp_dir):
        """Two generators with the same seed produce identical first output image."""
        out1 = tmp_dir / "rep_out1"
        out2 = tmp_dir / "rep_out2"
        SyntheticDefectGenerator(seed=42).add_cracks(
            str(sample_images_dir), str(out1), num_samples=1
        )
        SyntheticDefectGenerator(seed=42).add_cracks(
            str(sample_images_dir), str(out2), num_samples=1
        )
        imgs1 = sorted((out1 / "images").glob("*.jpg"))
        imgs2 = sorted((out2 / "images").glob("*.jpg"))
        assert len(imgs1) == 1 and len(imgs2) == 1
        arr1 = cv2.imread(str(imgs1[0]))
        arr2 = cv2.imread(str(imgs2[0]))
        assert np.array_equal(arr1, arr2), "Seeded outputs differ"


# ---------------------------------------------------------------------------
# IndustrialAugPipeline
# ---------------------------------------------------------------------------

class TestIndustrialAugPipeline:
    def test_pipeline_invalid_transform_raises(self):
        """Constructing with an unknown transform name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform"):
            IndustrialAugPipeline(["invalid"])

    def test_pipeline_run_creates_multiplied_images(self, sample_images_dir, tmp_dir):
        """5 source images × multiplier=2 → 10 augmented images."""
        out = tmp_dir / "aug_out"
        pipeline = IndustrialAugPipeline(["random_lighting", "industrial_noise"])
        pipeline.run(str(sample_images_dir), str(out), multiplier=2)
        images = list((out / "images").glob("*.jpg"))
        assert len(images) == 10

    def test_pipeline_motion_blur(self, tmp_dir):
        """Motion-blurred output differs from a non-uniform input image."""
        # Build a non-uniform source image (gradient) so blur is detectable
        non_uniform_dir = tmp_dir / "nu_src"
        non_uniform_dir.mkdir()
        gradient = np.tile(np.arange(640, dtype=np.uint8), (480, 1))
        img_bgr = cv2.merge([gradient, gradient, gradient])
        src_path = non_uniform_dir / "gradient.jpg"
        cv2.imwrite(str(src_path), img_bgr)

        out = tmp_dir / "mb_out"
        pipeline = IndustrialAugPipeline(["motion_blur"])
        pipeline.run(str(non_uniform_dir), str(out), multiplier=1)
        out_imgs = sorted((out / "images").glob("*.jpg"))
        assert len(out_imgs) >= 1
        src = cv2.imread(str(src_path))
        aug = cv2.imread(str(out_imgs[0]))
        assert not np.array_equal(src, aug), "Motion blur output identical to input"

    def test_pipeline_jpeg_compression(self, sample_images_dir, tmp_dir):
        """JPEG-compressed output file is created and is a valid JPEG."""
        out = tmp_dir / "jc_out"
        pipeline = IndustrialAugPipeline(["jpeg_compression"])
        pipeline.run(str(sample_images_dir), str(out), multiplier=1)
        out_imgs = list((out / "images").glob("*.jpg"))
        assert len(out_imgs) >= 1
        img = cv2.imread(str(out_imgs[0]))
        assert img is not None, "Output is not a readable JPEG"
