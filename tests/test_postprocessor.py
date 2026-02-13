"""
Tests for src/postprocessor.py — Export text, JSON, annotated images.
"""

import json
import numpy as np
import pytest

from src.postprocessor import Postprocessor


class TestMergeLines:
    def test_empty(self):
        post = Postprocessor()
        assert post.merge_lines([]) == []

    def test_single_result(self, sample_ocr_results):
        post = Postprocessor()
        lines = post.merge_lines(sample_ocr_results[:1])
        assert len(lines) == 1
        assert lines[0] == "Hello World"

    def test_multi_line(self, sample_ocr_results):
        post = Postprocessor()
        lines = post.merge_lines(sample_ocr_results)
        assert len(lines) == 3


class TestExportText:
    def test_creates_file(self, temp_output_dir, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_text(sample_ocr_results, "test_doc.jpg")

        assert path.exists()
        assert path.suffix == ".txt"
        assert "test_doc" in path.stem

    def test_content(self, temp_output_dir, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_text(sample_ocr_results, "test_doc.jpg")

        content = path.read_text(encoding="utf-8")
        assert "Hello World" in content
        assert "Second line" in content


class TestExportJSON:
    def test_creates_file(self, temp_output_dir, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_json(sample_ocr_results, "test_doc.jpg",
                                processing_time=1.5, device="cpu", mode="fast")

        assert path.exists()
        assert path.suffix == ".json"

    def test_json_structure(self, temp_output_dir, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_json(sample_ocr_results, "test_doc.jpg",
                                processing_time=1.5, device="cpu", mode="fast")

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["file"] == "test_doc.jpg"
        assert data["device"] == "cpu"
        assert data["mode"] == "fast"
        assert data["num_regions"] == 3
        assert "text" in data
        assert "regions" in data
        assert "metadata" in data
        assert len(data["regions"]) == 3

    def test_json_metadata(self, temp_output_dir, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_json(sample_ocr_results, "test_doc.jpg",
                                processing_time=1.5)

        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        assert meta["processing_time_seconds"] == 1.5
        assert 0 < meta["avg_confidence"] <= 1
        assert meta["status"] == "success"

    def test_json_no_results(self, temp_output_dir):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_json([], "empty.jpg")

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["num_regions"] == 0
        assert data["metadata"]["status"] == "no_text"


class TestExportAnnotated:
    def test_creates_file(self, temp_output_dir, sample_image, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_annotated(sample_image, sample_ocr_results, "test_doc.jpg")

        assert path.exists()
        assert path.suffix == ".png"

    def test_with_custom_draw_func(self, temp_output_dir, sample_image, sample_ocr_results):
        def custom_draw(image, results):
            return image.copy()

        post = Postprocessor(output_dir=str(temp_output_dir))
        path = post.export_annotated(sample_image, sample_ocr_results,
                                     "test_doc.jpg", draw_func=custom_draw)
        assert path.exists()


class TestExportAll:
    def test_returns_three_keys(self, temp_output_dir, sample_image, sample_ocr_results):
        post = Postprocessor(output_dir=str(temp_output_dir))
        exports = post.export_all(
            results=sample_ocr_results,
            image=sample_image,
            filename="test_doc.jpg",
            processing_time=0.5,
            device="cpu",
            mode="balanced",
        )

        assert "text" in exports
        assert "json" in exports
        assert "annotated" in exports
        assert exports["text"].exists()
        assert exports["json"].exists()
        assert exports["annotated"].exists()
