"""Tests for hf_offline/bundle.py orchestration and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: fake objects
# ---------------------------------------------------------------------------

def _fake_space_info(sdk="docker", port=8080, models=None):
    from hf_offline.space import SpaceInfo
    return SpaceInfo(
        space_id="hysts/translategemma-27b-it",
        owner="hysts",
        name="translategemma-27b-it",
        sdk=sdk,
        port=port,
        models=models if models is not None else ["google/gemma-2-27b-it"],
        title="TranslateGemma 27B IT",
        has_dockerfile=True,
    )


def _fake_model_info(model_id="google/gemma-2-27b-it"):
    from hf_offline.models import ModelInfo
    owner, name = model_id.split("/", 1)
    return ModelInfo(
        model_id=model_id,
        owner=owner,
        name=name,
        pipeline_tag="text2text-generation",
        library_name="transformers",
        license="gemma",
        tags=["translation"],
        gated=True,
        size_bytes=54_000_000_000,
        files=["config.json", "model.safetensors"],
        card_text="# Gemma\n\nA model.",
        card_meta={"license": "gemma"},
    )


def _fake_render_html(model_id, card_text, card_meta, space_id="", output_path=None):
    """Mock render_model_card_html that actually writes the file."""
    html = "<html><body>Model Card</body></html>"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    return html


def _bundle_patches(app_dir, space_info, model_info=None):
    """Return the common patch context managers for create_bundle tests."""
    if model_info is None:
        model_info = _fake_model_info()

    return {
        "hf_offline.bundle.clone_space": MagicMock(return_value=(app_dir, space_info)),
        "hf_offline.bundle.generate_dockerfile": MagicMock(),
        "hf_offline.bundle.download_model": MagicMock(return_value=(app_dir, model_info)),
        "hf_offline.bundle.write_model_manifest": MagicMock(),
        "hf_offline.bundle.export_space_images": MagicMock(return_value=[]),
        "hf_offline.bundle.render_model_card_html": MagicMock(side_effect=_fake_render_html),
    }


def _run_bundle(tmp_path, space_info=None, model_info=None,
                model_ids=None, output_dir=None, skip_images=True, **kwargs):
    """Run create_bundle with all HF calls mocked."""
    if space_info is None:
        space_info = _fake_space_info()

    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "Dockerfile").write_text("FROM python:3.10\n")

    out = output_dir or str(tmp_path / "bundle-hysts--translategemma-27b-it")
    patches = _bundle_patches(app_dir, space_info, model_info)

    with patch("hf_offline.bundle.clone_space", patches["hf_offline.bundle.clone_space"]), \
         patch("hf_offline.bundle.generate_dockerfile", patches["hf_offline.bundle.generate_dockerfile"]), \
         patch("hf_offline.bundle.download_model", patches["hf_offline.bundle.download_model"]), \
         patch("hf_offline.bundle.write_model_manifest", patches["hf_offline.bundle.write_model_manifest"]), \
         patch("hf_offline.bundle.export_space_images", patches["hf_offline.bundle.export_space_images"]), \
         patch("hf_offline.bundle.render_model_card_html", patches["hf_offline.bundle.render_model_card_html"]):

        from hf_offline.bundle import create_bundle
        result = create_bundle(
            "hysts/translategemma-27b-it",
            model_ids=model_ids or ["google/gemma-2-27b-it"],
            hf_token="hf_test_token",
            output_dir=out,
            skip_images=skip_images,
            **kwargs,
        )

    return result, patches


# ===========================================================================
# create_bundle - output structure
# ===========================================================================

class TestCreateBundleOutput:
    def test_returns_path(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert isinstance(result, Path)

    def test_output_dir_created(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert result.is_dir()

    def test_manifest_created(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert (result / "manifest.json").exists()

    def test_chart_dir_created(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert (result / "chart").is_dir()

    def test_chart_yaml_inside_chart_dir(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert (result / "chart" / "Chart.yaml").exists()

    def test_model_card_html_created(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        assert (result / "model-card.html").exists()


# ===========================================================================
# create_bundle - manifest content
# ===========================================================================

class TestManifestContent:
    def _manifest(self, tmp_path, **kwargs):
        result, _ = _run_bundle(tmp_path, **kwargs)
        return json.loads((result / "manifest.json").read_text())

    def test_manifest_version(self, tmp_path):
        assert self._manifest(tmp_path)["version"] == "1.0"

    def test_manifest_tool_field(self, tmp_path):
        assert self._manifest(tmp_path)["tool"] == "hf-offline"

    def test_manifest_space_id(self, tmp_path):
        assert self._manifest(tmp_path)["space"]["id"] == "hysts/translategemma-27b-it"

    def test_manifest_space_sdk(self, tmp_path):
        assert self._manifest(tmp_path)["space"]["sdk"] == "docker"

    def test_manifest_space_port(self, tmp_path):
        assert self._manifest(tmp_path)["space"]["port"] == 8080

    def test_manifest_model_entry(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["id"] == "google/gemma-2-27b-it"

    def test_manifest_model_path_uses_double_dash(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert manifest["models"][0]["path"] == "models/models--google--gemma-2-27b-it"

    def test_manifest_deploy_steps_present(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert len(manifest["deploy_steps"]) > 0

    def test_manifest_created_timestamp(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert "created" in manifest
        assert "T" in manifest["created"]  # ISO 8601

    def test_manifest_contents_section(self, tmp_path):
        manifest = self._manifest(tmp_path)
        contents = manifest["contents"]
        assert contents["app"] == "app/"
        assert contents["models"] == "models/"
        assert contents["chart"] == "chart/"

    def test_manifest_images_empty_when_skipped(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert manifest["images"] == []

    def test_manifest_is_valid_json(self, tmp_path):
        result, _ = _run_bundle(tmp_path)
        text = (result / "manifest.json").read_text()
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


# ===========================================================================
# _human_size helper
# ===========================================================================

class TestHumanSize:
    def _size(self, n):
        from hf_offline.bundle import _human_size
        return _human_size(n)

    def test_bytes(self):
        assert self._size(500) == "500.0 B"

    def test_kilobytes(self):
        assert self._size(1024) == "1.0 KB"

    def test_megabytes(self):
        assert self._size(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert self._size(1024 ** 3) == "1.0 GB"

    def test_terabytes(self):
        assert self._size(1024 ** 4) == "1.0 TB"

    def test_large_model_size_is_gb(self):
        # 54 GB (roughly gemma-2-27b-it)
        result = self._size(54 * 1024 ** 3)
        assert "GB" in result

    def test_zero_bytes(self):
        assert self._size(0) == "0.0 B"

    def test_fractional_mb(self):
        result = self._size(int(1.5 * 1024 * 1024))
        assert "MB" in result
        assert "1.5" in result


# ===========================================================================
# create_bundle - step invocations
# ===========================================================================

class TestCreateBundleSteps:
    def test_clone_space_called_once(self, tmp_path):
        _, patches = _run_bundle(tmp_path)
        patches["hf_offline.bundle.clone_space"].assert_called_once()

    def test_clone_space_called_with_space_id(self, tmp_path):
        _, patches = _run_bundle(tmp_path)
        call_args = patches["hf_offline.bundle.clone_space"].call_args
        assert "hysts/translategemma-27b-it" in call_args[0]

    def test_download_model_called_per_model(self, tmp_path):
        space_info = _fake_space_info(models=["model/a", "model/b"])
        _, patches = _run_bundle(tmp_path, space_info=space_info, model_ids=["model/a", "model/b"])
        assert patches["hf_offline.bundle.download_model"].call_count == 2

    def test_download_model_called_with_token(self, tmp_path):
        _, patches = _run_bundle(tmp_path)
        call_kwargs = patches["hf_offline.bundle.download_model"].call_args[1]
        assert call_kwargs["token"] == "hf_test_token"

    def test_model_download_failure_continues(self, tmp_path):
        """A single model download failure should not abort the whole bundle."""
        space_info = _fake_space_info(models=["model/a", "model/b"])
        app_dir = tmp_path / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "Dockerfile").write_text("FROM python:3.10\n")

        call_count = [0]
        def dl_side_effect(model_id, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("network error")
            return (app_dir, _fake_model_info(model_id))

        with patch("hf_offline.bundle.clone_space", return_value=(app_dir, space_info)), \
             patch("hf_offline.bundle.generate_dockerfile"), \
             patch("hf_offline.bundle.download_model", side_effect=dl_side_effect), \
             patch("hf_offline.bundle.write_model_manifest"), \
             patch("hf_offline.bundle.export_space_images", return_value=[]), \
             patch("hf_offline.bundle.render_model_card_html", side_effect=_fake_render_html):

            from hf_offline.bundle import create_bundle
            result = create_bundle(
                "hysts/translategemma-27b-it",
                model_ids=["model/a", "model/b"],
                output_dir=str(tmp_path / "out"),
                skip_images=True,
            )

        # Should complete despite one failure
        assert result.is_dir()
        assert (result / "manifest.json").exists()

    def test_export_images_not_called_when_skipped(self, tmp_path):
        _, patches = _run_bundle(tmp_path, skip_images=True)
        patches["hf_offline.bundle.export_space_images"].assert_not_called()

    def test_export_images_called_when_not_skipped(self, tmp_path):
        _, patches = _run_bundle(tmp_path, skip_images=False)
        patches["hf_offline.bundle.export_space_images"].assert_called_once()

    def test_write_model_manifest_called_per_model(self, tmp_path):
        _, patches = _run_bundle(tmp_path)
        patches["hf_offline.bundle.write_model_manifest"].assert_called_once()

    def test_generate_dockerfile_called(self, tmp_path):
        _, patches = _run_bundle(tmp_path)
        patches["hf_offline.bundle.generate_dockerfile"].assert_called_once()

    def test_safe_name_uses_double_dash(self, tmp_path):
        """Default output dir should use -- to replace / in space_id."""
        import os
        app_dir = tmp_path / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "Dockerfile").write_text("FROM python:3.10\n")

        space_info = _fake_space_info()

        with patch("hf_offline.bundle.clone_space", return_value=(app_dir, space_info)), \
             patch("hf_offline.bundle.generate_dockerfile"), \
             patch("hf_offline.bundle.download_model", return_value=(app_dir, _fake_model_info())), \
             patch("hf_offline.bundle.write_model_manifest"), \
             patch("hf_offline.bundle.export_space_images", return_value=[]), \
             patch("hf_offline.bundle.render_model_card_html", side_effect=_fake_render_html):

            orig = os.getcwd()
            os.chdir(tmp_path)
            try:
                from hf_offline.bundle import create_bundle
                result = create_bundle(
                    "hysts/translategemma-27b-it",
                    model_ids=["google/gemma-2-27b-it"],
                    skip_images=True,
                )
            finally:
                os.chdir(orig)

        assert "hysts--translategemma-27b-it" in result.name
        assert "/" not in result.name

    def test_no_models_message_shown_not_crash(self, tmp_path):
        """When no models detected, bundle should still complete."""
        space_info = _fake_space_info(models=[])
        result, _ = _run_bundle(tmp_path, space_info=space_info, model_ids=[])
        assert result.is_dir()
