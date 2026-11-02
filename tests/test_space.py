"""Comprehensive tests for hf_offline/space.py."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _space_info_raw(
    sdk="docker",
    models=None,
    tags=None,
):
    """Build a fake HF Hub SpaceInfo object."""
    return SimpleNamespace(
        id="hysts/translategemma-27b-it",
        sdk=sdk,
        sdk_version=None,
        models=models if models is not None else ["google/gemma-2-27b-it"],
        tags=tags or ["translation", "gemma"],
        cardData=None,
        runtime=None,
    )


def _make_api(tmp_readme: Path, space_info_raw=None):
    if space_info_raw is None:
        space_info_raw = _space_info_raw()
    api = MagicMock()
    api.space_info.return_value = space_info_raw
    api.hf_hub_download.return_value = str(tmp_readme)
    return api


DOCKER_README = textwrap.dedent("""\
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
    """)

GRADIO_README = textwrap.dedent("""\
    ---
    title: My Gradio Space
    sdk: gradio
    sdk_version: 4.0.0
    python_version: "3.11"
    app_file: demo.py
    models:
      - google/gemma-2-9b-it
    ---
    """)


# ===========================================================================
# fetch_space_metadata
# ===========================================================================

class TestFetchSpaceMetadata:
    def test_docker_sdk_parsed(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(DOCKER_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert info.sdk == "docker"

    def test_gradio_sdk_parsed(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(GRADIO_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(sdk="gradio", models=["google/gemma-2-9b-it"]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.sdk == "gradio"

    def test_models_from_api_included(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(DOCKER_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=["google/gemma-2-27b-it"]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert "google/gemma-2-27b-it" in info.models

    def test_models_from_readme_merged(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(textwrap.dedent("""\
            ---
            sdk: docker
            models:
              - extra/model
            ---
            """))
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=["api/model"]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert "api/model" in info.models
        assert "extra/model" in info.models

    def test_models_deduplicated(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(textwrap.dedent("""\
            ---
            sdk: docker
            models:
              - google/gemma-2-27b-it
            ---
            """))
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=["google/gemma-2-27b-it"]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert info.models.count("google/gemma-2-27b-it") == 1

    def test_api_models_listed_first(self, tmp_path):
        """API models should come before README-only models."""
        readme = tmp_path / "README.md"
        readme.write_text(textwrap.dedent("""\
            ---
            sdk: docker
            models:
              - readme/model
            ---
            """))
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=["api/model"]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert info.models[0] == "api/model"

    def test_no_models_returns_empty_list(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("---\nsdk: gradio\n---\n")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.models == []

    def test_readme_download_failure_uses_api_models(self, tmp_path):
        api = MagicMock()
        api.space_info.return_value = _space_info_raw(models=["google/gemma-2-27b-it"])
        api.hf_hub_download.side_effect = Exception("403 Forbidden")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = api
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert "google/gemma-2-27b-it" in info.models

    def test_port_docker(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(DOCKER_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert info.port == 8080

    def test_port_gradio(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(GRADIO_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(sdk="gradio", models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.port == 7860

    def test_port_streamlit(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("---\nsdk: streamlit\n---\n")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(sdk="streamlit", models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.port == 8501

    def test_python_version_from_readme(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(GRADIO_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(sdk="gradio", models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.python_version == "3.11"

    def test_python_version_defaults_to_310(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("---\nsdk: gradio\n---\n")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(sdk="gradio", models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert info.python_version == "3.10"

    def test_owner_and_name_split(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(DOCKER_README)
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme)
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert info.owner == "hysts"
        assert info.name == "translategemma-27b-it"

    def test_models_none_from_api_handled(self, tmp_path):
        """space_info_raw.models can be None - should not crash."""
        readme = tmp_path / "README.md"
        readme.write_text("---\nsdk: docker\n---\n")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=None))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("hysts/translategemma-27b-it")
        assert isinstance(info.models, list)

    def test_single_model_string_in_readme(self, tmp_path):
        """models: some/model (not a list) should be normalised to a list."""
        readme = tmp_path / "README.md"
        readme.write_text("---\nsdk: docker\nmodels: some/single-model\n---\n")
        with patch("hf_offline.space.HfApi") as M:
            M.return_value = _make_api(readme, _space_info_raw(models=[]))
            from hf_offline.space import fetch_space_metadata
            info = fetch_space_metadata("test/space")
        assert "some/single-model" in info.models


# ===========================================================================
# _parse_yaml_frontmatter
# ===========================================================================

class TestParseYamlFrontmatter:
    def _parse(self, text):
        from hf_offline.space import _parse_yaml_frontmatter
        return _parse_yaml_frontmatter(text)

    def test_valid_frontmatter(self):
        text = "---\ntitle: Test\nsdk: gradio\n---\nBody text"
        result = self._parse(text)
        assert result["title"] == "Test"
        assert result["sdk"] == "gradio"

    def test_no_frontmatter_returns_empty(self):
        assert self._parse("Just body text") == {}

    def test_empty_frontmatter_returns_empty(self):
        assert self._parse("---\n---\nBody") == {}

    def test_invalid_yaml_returns_empty(self):
        assert self._parse("---\n: bad: yaml: [\n---\nBody") == {}

    def test_multiline_frontmatter(self):
        text = "---\ntitle: Long Title\nmodels:\n  - org/model-a\n  - org/model-b\n---\n"
        result = self._parse(text)
        assert result["models"] == ["org/model-a", "org/model-b"]

    def test_frontmatter_with_emoji(self):
        text = "---\ntitle: Space 🌍\n---\nBody"
        result = self._parse(text)
        assert result["title"] == "Space 🌍"


# ===========================================================================
# scan_for_model_refs
# ===========================================================================

class TestScanForModelRefs:
    def _scan(self, tmp_path, code, filename="app.py"):
        (tmp_path / filename).write_text(code)
        from hf_offline.space import scan_for_model_refs
        return scan_for_model_refs(tmp_path)

    def test_from_pretrained_single_quotes(self, tmp_path):
        refs = self._scan(tmp_path, "m = AutoModel.from_pretrained('google/gemma-2-27b-it')\n")
        assert "google/gemma-2-27b-it" in refs

    def test_from_pretrained_double_quotes(self, tmp_path):
        refs = self._scan(tmp_path, 'm = AutoModel.from_pretrained("google/gemma-2-27b-it")\n')
        assert "google/gemma-2-27b-it" in refs

    def test_pipeline_model_kwarg(self, tmp_path):
        refs = self._scan(tmp_path, 'pipe = pipeline("translation", model="google/gemma-2-27b-it")\n')
        assert "google/gemma-2-27b-it" in refs

    def test_hf_hub_download_repo_id(self, tmp_path):
        refs = self._scan(tmp_path, 'hf_hub_download(repo_id="google/gemma-2-27b-it", filename="config.json")\n')
        assert "google/gemma-2-27b-it" in refs

    def test_snapshot_download_repo_id(self, tmp_path):
        refs = self._scan(tmp_path, 'snapshot_download(repo_id="google/gemma-2-27b-it", local_dir="/m")\n')
        assert "google/gemma-2-27b-it" in refs

    def test_model_info_repo_id(self, tmp_path):
        refs = self._scan(tmp_path, 'model_info(repo_id="google/gemma-2-27b-it")\n')
        assert "google/gemma-2-27b-it" in refs

    def test_multiline_hf_hub_download(self, tmp_path):
        code = textwrap.dedent("""\
            hf_hub_download(
                repo_id="google/gemma-2-27b-it",
                filename="config.json",
            )
            """)
        refs = self._scan(tmp_path, code)
        assert "google/gemma-2-27b-it" in refs

    def test_multiline_snapshot_download(self, tmp_path):
        code = textwrap.dedent("""\
            snapshot_download(
                repo_id="google/gemma-2-27b-it",
                local_dir="/models",
            )
            """)
        refs = self._scan(tmp_path, code)
        assert "google/gemma-2-27b-it" in refs

    def test_generic_space_repo_id_not_included(self, tmp_path):
        refs = self._scan(tmp_path, 'api.space_info(repo_id="hysts/translategemma-27b-it")\n')
        assert "hysts/translategemma-27b-it" not in refs

    def test_false_positive_gradio_filtered(self, tmp_path):
        refs = self._scan(tmp_path, '# gradio/gradio test\n')
        assert "gradio/gradio" not in refs

    def test_false_positive_numpy_filtered(self, tmp_path):
        refs = self._scan(tmp_path, 'import numpy\n# numpy/numpy\n')
        assert "numpy/numpy" not in refs

    def test_multiple_models_in_file(self, tmp_path):
        code = textwrap.dedent("""\
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
            model = AutoModel.from_pretrained("google/gemma-2-9b-it")
            """)
        refs = self._scan(tmp_path, code)
        assert "google/gemma-2-27b-it" in refs
        assert "google/gemma-2-9b-it" in refs

    def test_returns_sorted_unique_list(self, tmp_path):
        code = textwrap.dedent("""\
            m1 = AutoModel.from_pretrained("b/model")
            m2 = AutoModel.from_pretrained("a/model")
            m3 = AutoModel.from_pretrained("a/model")
            """)
        refs = self._scan(tmp_path, code)
        assert refs == sorted(set(refs))
        assert refs.count("a/model") == 1

    def test_scans_nested_py_files(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "utils.py").write_text('m = AutoModel.from_pretrained("org/nested-model")\n')
        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "org/nested-model" in refs

    def test_non_py_files_ignored(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model": "org/json-model"}')
        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "org/json-model" not in refs

    def test_invalid_file_encoding_skipped(self, tmp_path):
        """Files with bad encoding should not crash the scan."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_bytes(b'\xff\xfe invalid bytes \xff')
        (tmp_path / "good.py").write_text('m = AutoModel.from_pretrained("org/good-model")\n')
        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert "org/good-model" in refs

    def test_empty_directory(self, tmp_path):
        from hf_offline.space import scan_for_model_refs
        refs = scan_for_model_refs(tmp_path)
        assert refs == []


# ===========================================================================
# generate_dockerfile
# ===========================================================================

class TestGenerateDockerfile:
    def _info(self, sdk, has_dockerfile=False, has_requirements=False,
               python_version="3.10", app_file="app.py", port=None):
        from hf_offline.space import SpaceInfo, SDK_PORTS
        return SpaceInfo(
            space_id=f"test/{sdk}-space",
            owner="test",
            name=f"{sdk}-space",
            sdk=sdk,
            has_dockerfile=has_dockerfile,
            has_requirements=has_requirements,
            python_version=python_version,
            app_file=app_file,
            port=port or SDK_PORTS.get(sdk, 7860),
        )

    def test_existing_dockerfile_preserved(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        existing = app_dir / "Dockerfile"
        existing.write_text("FROM my-custom-image\n")
        from hf_offline.space import generate_dockerfile
        generate_dockerfile(self._info("docker", has_dockerfile=True), app_dir)
        assert existing.read_text() == "FROM my-custom-image\n"

    def test_gradio_dockerfile_base_image(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("gradio", python_version="3.11"), app_dir)
        assert "FROM python:3.11-slim" in df.read_text()

    def test_gradio_dockerfile_exposes_correct_port(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("gradio"), app_dir)
        assert "EXPOSE 7860" in df.read_text()

    def test_gradio_dockerfile_with_requirements(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "requirements.txt").write_text("gradio\n")
        info = self._info("gradio", has_requirements=True)
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(info, app_dir)
        assert "requirements.txt" in df.read_text()
        assert "pip install" in df.read_text()

    def test_gradio_dockerfile_without_requirements_installs_gradio(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("gradio", has_requirements=False), app_dir)
        assert "pip install" in df.read_text()
        assert "gradio" in df.read_text()

    def test_streamlit_dockerfile_base_image(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("streamlit", python_version="3.10"), app_dir)
        assert "FROM python:3.10-slim" in df.read_text()

    def test_streamlit_dockerfile_exposes_port(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("streamlit"), app_dir)
        assert "EXPOSE 8501" in df.read_text()

    def test_streamlit_cmd_uses_app_file(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("streamlit", app_file="dashboard.py"), app_dir)
        assert "dashboard.py" in df.read_text()

    def test_generic_dockerfile_for_unknown_sdk(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("static"), app_dir)
        assert df.exists()

    def test_gradio_cmd_uses_app_file(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        from hf_offline.space import generate_dockerfile
        df = generate_dockerfile(self._info("gradio", app_file="my_app.py"), app_dir)
        assert "my_app.py" in df.read_text()


class TestDockerfileInstallBlock:
    def _gen(self, tmp_path, sdk="gradio", ver="5.1.0", reqs=None, pkgs=None):
        from hf_offline.space import SpaceInfo, generate_dockerfile

        if reqs is not None:
            (tmp_path / "requirements.txt").write_text(reqs)
        if pkgs is not None:
            (tmp_path / "packages.txt").write_text(pkgs)
        info = SpaceInfo(space_id="o/s", owner="o", name="s", sdk=sdk, sdk_version=ver,
                         has_requirements=reqs is not None)
        return generate_dockerfile(info, tmp_path).read_text()

    def test_apt_packages_installed(self, tmp_path):
        d = self._gen(tmp_path, reqs="pytesseract\n", pkgs="tesseract-ocr\n# c\npoppler-utils\n")
        assert "apt-get install" in d and "COPY packages.txt" in d

    def test_sdk_pinned_when_missing_from_requirements(self, tmp_path):
        assert "pip install --no-cache-dir gradio==5.1.0" in self._gen(tmp_path, reqs="pillow\n")

    def test_sdk_not_duplicated_when_in_requirements(self, tmp_path):
        assert "gradio==" not in self._gen(tmp_path, reqs="Gradio>=4\n")

    def test_no_packages_no_apt(self, tmp_path):
        assert "apt-get" not in self._gen(tmp_path, reqs="pillow\n")
