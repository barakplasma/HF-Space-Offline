"""Tests simulating hysts/translategemma-27b-it space bundling.

All HuggingFace Hub API calls are mocked so these tests run offline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures: fake HF API responses for hysts/translategemma-27b-it
# ---------------------------------------------------------------------------

SPACE_README = textwrap.dedent("""\
    ---
    title: TranslateGemma 27B IT
    emoji: 🌍
    sdk: docker
    pinned: false
    license: gemma
    tags:
      - translation
      - gemma
    ---

    # TranslateGemma 27B IT

    A translation demo using Google's Gemma 27B instruction-tuned model.
    """)

FAKE_SPACE_INFO = SimpleNamespace(
    id="hysts/translategemma-27b-it",
    sdk="docker",
    tags=["translation", "gemma"],
    models=["google/gemma-2-27b-it"],  # <-- provided by HF Hub API
    cardData=None,
)

FAKE_MODEL_INFO = SimpleNamespace(
    id="google/gemma-2-27b-it",
    pipeline_tag="text2text-generation",
    library_name="transformers",
    tags=["translation", "gemma", "text2text-generation"],
    gated="auto",
)

MODEL_CARD_TEXT = textwrap.dedent("""\
    ---
    license: gemma
    pipeline_tag: text2text-generation
    library_name: transformers
    ---

    # Gemma 2 27B IT

    Google's Gemma 2 27B instruction-tuned model for translation tasks.
    """)


# ---------------------------------------------------------------------------
# Tests: space metadata fetching
# ---------------------------------------------------------------------------

class TestFetchSpaceMetadata:
    """Tests for fetch_space_metadata with translategemma-27b-it."""

    def _make_mock_api(self, tmp_readme: Path):
        api = MagicMock()
        api.space_info.return_value = FAKE_SPACE_INFO
        api.hf_hub_download.return_value = str(tmp_readme)
        return api

    def test_sdk_is_docker(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(SPACE_README)

        with patch("hf_offline.space.HfApi") as MockApi:
            MockApi.return_value = self._make_mock_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")

        assert info.sdk == "docker"

    def test_models_from_api_included(self, tmp_path):
        """Models listed in the HF Hub API response must appear in the result."""
        readme = tmp_path / "README.md"
        readme.write_text(SPACE_README)

        with patch("hf_offline.space.HfApi") as MockApi:
            MockApi.return_value = self._make_mock_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")

        assert "google/gemma-2-27b-it" in info.models

    def test_models_deduplicated(self, tmp_path):
        """If a model appears in both API and README, it should not be duplicated."""
        readme_with_model = textwrap.dedent("""\
            ---
            title: Test
            sdk: docker
            models:
              - google/gemma-2-27b-it
            ---
            """)
        readme = tmp_path / "README.md"
        readme.write_text(readme_with_model)

        with patch("hf_offline.space.HfApi") as MockApi:
            MockApi.return_value = self._make_mock_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")

        assert info.models.count("google/gemma-2-27b-it") == 1

    def test_port_for_docker_sdk(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(SPACE_README)

        with patch("hf_offline.space.HfApi") as MockApi:
            MockApi.return_value = self._make_mock_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")

        assert info.port == 8080

    def test_readme_parse_failure_does_not_crash(self, tmp_path):
        """If README download fails, metadata should still be returned."""
        api = MagicMock()
        api.space_info.return_value = FAKE_SPACE_INFO
        api.hf_hub_download.side_effect = Exception("network error")

        with patch("hf_offline.space.HfApi") as MockApi:
            MockApi.return_value = api
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")

        # Should still include models from the API response
        assert "google/gemma-2-27b-it" in info.models


# ---------------------------------------------------------------------------
# Tests: scan_for_model_refs
# ---------------------------------------------------------------------------

class TestScanForModelRefs:
    """Tests for model reference scanning in space code."""

    def test_from_pretrained_detected(self, tmp_path):
        code = 'model = AutoModelForSeq2SeqLM.from_pretrained("google/gemma-2-27b-it")\n'
        (tmp_path / "app.py").write_text(code)

        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "google/gemma-2-27b-it" in refs

    def test_model_kwarg_detected(self, tmp_path):
        code = 'pipe = pipeline("translation", model="google/gemma-2-27b-it")\n'
        (tmp_path / "app.py").write_text(code)

        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "google/gemma-2-27b-it" in refs

    def test_generic_repo_id_not_included(self, tmp_path):
        """repo_id= in a non-model context should not be added."""
        code = 'api.space_info(repo_id="hysts/translategemma-27b-it")\n'
        (tmp_path / "app.py").write_text(code)

        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        # space repo_id should not be treated as a model
        assert "hysts/translategemma-27b-it" not in refs

    def test_hf_hub_download_repo_id_detected(self, tmp_path):
        code = 'hf_hub_download(repo_id="google/gemma-2-27b-it", filename="config.json")\n'
        (tmp_path / "app.py").write_text(code)

        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "google/gemma-2-27b-it" in refs

    def test_false_positive_libraries_filtered(self, tmp_path):
        code = 'import gradio/gradio\n'
        (tmp_path / "app.py").write_text(code)

        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "gradio/gradio" not in refs


# ---------------------------------------------------------------------------
# Tests: generate_dockerfile  (Docker SDK spaces keep existing Dockerfile)
# ---------------------------------------------------------------------------

class TestGenerateDockerfile:
    """Tests for Dockerfile generation logic."""

    def test_docker_sdk_keeps_existing_dockerfile(self, tmp_path):
        """Spaces using Docker SDK already have a Dockerfile - don't overwrite."""
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        existing = app_dir / "Dockerfile"
        existing.write_text("FROM custom-base\n")

        from hf_offline.space import SpaceInfo, generate_dockerfile

        info = SpaceInfo(
            space_id="hysts/translategemma-27b-it",
            owner="hysts",
            name="translategemma-27b-it",
            sdk="docker",
            has_dockerfile=True,
        )
        generate_dockerfile(info, app_dir)
        assert existing.read_text() == "FROM custom-base\n"

    def test_gradio_dockerfile_generated(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()

        from hf_offline.space import SpaceInfo, generate_dockerfile

        info = SpaceInfo(
            space_id="test/space",
            owner="test",
            name="space",
            sdk="gradio",
            has_dockerfile=False,
            has_requirements=False,
            python_version="3.11",
        )
        df_path = generate_dockerfile(info, app_dir)
        content = df_path.read_text()
        assert "FROM python:3.11-slim" in content
        assert "gradio" in content.lower()


# ---------------------------------------------------------------------------
# Tests: write_model_manifest  (per-model, no overwrite)
# ---------------------------------------------------------------------------

class TestWriteModelManifest:
    """Tests for model manifest writing."""

    def _make_model_info(self, model_id: str):
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
            size_bytes=54_000_000_000,
        )

    def test_manifest_written_in_model_subdir(self, tmp_path):
        from hf_offline.models import write_model_manifest

        info = self._make_model_info("google/gemma-2-27b-it")
        path = write_model_manifest(info, tmp_path)

        assert path == tmp_path / "models" / "google--gemma-2-27b-it" / "manifest.yaml"
        assert path.exists()

    def test_two_models_do_not_overwrite_each_other(self, tmp_path):
        from hf_offline.models import write_model_manifest

        info1 = self._make_model_info("google/gemma-2-27b-it")
        info2 = self._make_model_info("meta-llama/Llama-3-8b")
        info2.pipeline_tag = "text-generation"

        write_model_manifest(info1, tmp_path)
        write_model_manifest(info2, tmp_path)

        m1 = yaml.safe_load(
            (tmp_path / "models" / "google--gemma-2-27b-it" / "manifest.yaml").read_text()
        )
        m2 = yaml.safe_load(
            (tmp_path / "models" / "meta-llama--Llama-3-8b" / "manifest.yaml").read_text()
        )
        assert m1["model_id"] == "google/gemma-2-27b-it"
        assert m2["model_id"] == "meta-llama/Llama-3-8b"

    def test_manifest_contains_expected_fields(self, tmp_path):
        from hf_offline.models import write_model_manifest

        info = self._make_model_info("google/gemma-2-27b-it")
        info.gated = True
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())

        assert data["model_id"] == "google/gemma-2-27b-it"
        assert data["pipeline_tag"] == "text2text-generation"
        assert data["gated"] is True
        assert data["size_bytes"] == 54_000_000_000


# ---------------------------------------------------------------------------
# Tests: render_model_card_html  (no Jinja2 artifacts)
# ---------------------------------------------------------------------------

class TestRenderModelCardHtml:
    """Tests for model card HTML rendering."""

    def _render(self, space_id: str = "hysts/translategemma-27b-it"):
        from hf_offline.model_card import render_model_card_html

        card_meta = {
            "license": "gemma",
            "pipeline_tag": "text2text-generation",
            "library_name": "transformers",
            "tags": ["translation", "gemma"],
        }
        return render_model_card_html(
            model_id="google/gemma-2-27b-it",
            card_text="# Gemma 2 27B\n\nA translation model.",
            card_meta=card_meta,
            space_id=space_id,
        )

    def test_no_jinja2_artifacts_in_output(self):
        html = self._render()
        assert "{%" not in html
        assert "%}" not in html

    def test_space_ref_shown_when_provided(self):
        html = self._render(space_id="hysts/translategemma-27b-it")
        assert "hysts/translategemma-27b-it" in html
        assert "Bundled with Space" in html

    def test_space_ref_absent_when_empty(self):
        html = self._render(space_id="")
        assert "Bundled with Space" not in html

    def test_model_id_in_output(self):
        html = self._render()
        assert "google/gemma-2-27b-it" in html

    def test_pipeline_badge_present(self):
        html = self._render()
        assert "text2text-generation" in html

    def test_html_written_to_file(self, tmp_path):
        from hf_offline.model_card import render_model_card_html

        out = tmp_path / "model-card.html"
        render_model_card_html(
            model_id="google/gemma-2-27b-it",
            card_text="# Test",
            card_meta={"license": "gemma"},
            space_id="hysts/translategemma-27b-it",
            output_path=out,
        )
        assert out.exists()
        content = out.read_text()
        assert "google/gemma-2-27b-it" in content

    def test_html_escape_in_model_id(self):
        """Model IDs with special chars should be escaped."""
        from hf_offline.model_card import render_model_card_html

        html = render_model_card_html(
            model_id='evil<script>alert(1)</script>',
            card_text="",
            card_meta={},
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Tests: Helm chart generation for translategemma-27b-it
# ---------------------------------------------------------------------------

class TestHelmChartGeneration:
    """Tests for Helm chart generation."""

    def _make_space_info(self):
        from hf_offline.space import SpaceInfo
        return SpaceInfo(
            space_id="hysts/translategemma-27b-it",
            owner="hysts",
            name="translategemma-27b-it",
            sdk="docker",
            port=8080,
            models=["google/gemma-2-27b-it"],
            title="TranslateGemma 27B IT",
        )

    def test_chart_yaml_created(self, tmp_path):
        from hf_offline.helm import generate_chart

        chart_dir = generate_chart(
            self._make_space_info(),
            tmp_path,
            model_ids=["google/gemma-2-27b-it"],
            gpu_count=2,
        )
        assert (chart_dir / "Chart.yaml").exists()

    def test_chart_name_sanitized(self, tmp_path):
        from hf_offline.helm import generate_chart

        chart_dir = generate_chart(
            self._make_space_info(),
            tmp_path,
            model_ids=["google/gemma-2-27b-it"],
        )
        chart = yaml.safe_load((chart_dir / "Chart.yaml").read_text())
        # underscores and dots replaced with hyphens, lowercased
        assert "_" not in chart["name"]
        assert chart["name"] == "translategemma-27b-it"

    def test_values_yaml_has_correct_port(self, tmp_path):
        from hf_offline.helm import generate_chart

        chart_dir = generate_chart(
            self._make_space_info(),
            tmp_path,
            model_ids=["google/gemma-2-27b-it"],
        )
        values = yaml.safe_load((chart_dir / "values.yaml").read_text())
        assert values["space"]["port"] == 8080

    def test_values_yaml_has_gpu_config(self, tmp_path):
        from hf_offline.helm import generate_chart

        chart_dir = generate_chart(
            self._make_space_info(),
            tmp_path,
            model_ids=["google/gemma-2-27b-it"],
            gpu_count=2,
            gpu_type="nvidia.com/gpu",
        )
        values = yaml.safe_load((chart_dir / "values.yaml").read_text())
        assert values["gpu"]["count"] == 2
        assert values["gpu"]["type"] == "nvidia.com/gpu"

    def test_all_template_files_created(self, tmp_path):
        from hf_offline.helm import generate_chart

        chart_dir = generate_chart(
            self._make_space_info(),
            tmp_path,
            model_ids=["google/gemma-2-27b-it"],
        )
        expected = [
            "Chart.yaml", "values.yaml",
            "templates/_helpers.tpl",
            "templates/deployment.yaml",
            "templates/service.yaml",
            "templates/ingress.yaml",
            "templates/pvc.yaml",
            "templates/secret.yaml",
            "templates/configmap.yaml",
            "templates/NOTES.txt",
        ]
        for f in expected:
            assert (chart_dir / f).exists(), f"Missing: {f}"
