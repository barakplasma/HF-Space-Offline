"""Comprehensive tests for hf_offline/models.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HF_TOKEN = "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"

def _fake_raw_model(
    pipeline_tag="text2text-generation",
    library_name="transformers",
    tags=None,
    gated="auto",
):
    return SimpleNamespace(
        id="google/gemma-2-27b-it",
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        tags=tags or ["translation", "gemma"],
        gated=gated,
    )


def _fake_card(text="# Gemma\n\nA model.", meta=None):
    card = MagicMock()
    card.text = text
    card.data = MagicMock()
    card.data.to_dict.return_value = meta or {"license": "gemma", "pipeline_tag": "text2text-generation"}
    return card


def _make_model_info(
    model_id="google/gemma-2-27b-it",
    pipeline_tag="text2text-generation",
    gated=True,
    size_bytes=0,
    files=None,
):
    from hf_offline.models import ModelInfo
    owner, name = model_id.split("/", 1) if "/" in model_id else ("", model_id)
    return ModelInfo(
        model_id=model_id,
        owner=owner,
        name=name,
        pipeline_tag=pipeline_tag,
        library_name="transformers",
        license="gemma",
        tags=["translation"],
        gated=gated,
        size_bytes=size_bytes,
        files=files or [],
    )


# ===========================================================================
# fetch_model_info
# ===========================================================================

class TestFetchModelInfo:
    def _fetch(self, model_id="google/gemma-2-27b-it", gated="auto",
               pipeline_tag="text2text-generation", token=None):
        raw = _fake_raw_model(gated=gated, pipeline_tag=pipeline_tag)
        card = _fake_card()
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            from hf_offline.models import fetch_model_info
            return fetch_model_info(model_id, token=token)

    def test_model_id_stored(self):
        info = self._fetch("google/gemma-2-27b-it")
        assert info.model_id == "google/gemma-2-27b-it"

    def test_owner_and_name_split(self):
        info = self._fetch("google/gemma-2-27b-it")
        assert info.owner == "google"
        assert info.name == "gemma-2-27b-it"

    def test_model_without_slash(self):
        raw = _fake_raw_model()
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = _fake_card()
            from hf_offline.models import fetch_model_info
            info = fetch_model_info("localmodel")
        assert info.owner == ""
        assert info.name == "localmodel"

    def test_pipeline_tag_stored(self):
        info = self._fetch(pipeline_tag="text-generation")
        assert info.pipeline_tag == "text-generation"

    def test_gated_auto_becomes_true(self):
        info = self._fetch(gated="auto")
        assert info.gated is True

    def test_gated_false_string_becomes_false(self):
        info = self._fetch(gated="false")
        assert info.gated is False

    def test_gated_bool_false_becomes_false(self):
        info = self._fetch(gated=False)
        assert info.gated is False

    def test_gated_bool_true_becomes_true(self):
        info = self._fetch(gated=True)
        assert info.gated is True

    def test_card_text_stored(self):
        raw = _fake_raw_model()
        card = _fake_card(text="# Gemma Model Card")
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            from hf_offline.models import fetch_model_info
            info = fetch_model_info("google/gemma-2-27b-it")
        assert "Gemma Model Card" in info.card_text

    def test_card_load_failure_returns_empty_card(self):
        raw = _fake_raw_model()
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.side_effect = Exception("network error")
            from hf_offline.models import fetch_model_info
            info = fetch_model_info("google/gemma-2-27b-it")
        assert info.card_text == ""
        assert info.card_meta == {}

    def test_token_passed_to_api(self):
        raw = _fake_raw_model()
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = _fake_card()
            from hf_offline.models import fetch_model_info
            fetch_model_info("google/gemma-2-27b-it", token=HF_TOKEN)
        MockApi.assert_called_once_with(token=HF_TOKEN)

    def test_none_pipeline_tag_becomes_empty_string(self):
        raw = _fake_raw_model(pipeline_tag=None)
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = _fake_card()
            from hf_offline.models import fetch_model_info
            info = fetch_model_info("google/gemma-2-27b-it")
        assert info.pipeline_tag == ""

    def test_none_library_name_becomes_empty_string(self):
        raw = _fake_raw_model(library_name=None)
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = _fake_card()
            from hf_offline.models import fetch_model_info
            info = fetch_model_info("google/gemma-2-27b-it")
        assert info.library_name == ""


# ===========================================================================
# write_model_manifest
# ===========================================================================

class TestWriteModelManifest:
    def test_manifest_in_per_model_subdir(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("google/gemma-2-27b-it")
        path = write_model_manifest(info, tmp_path)
        assert path == tmp_path / "models" / "models--google--gemma-2-27b-it" / "manifest.yaml"

    def test_manifest_file_exists(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("google/gemma-2-27b-it")
        path = write_model_manifest(info, tmp_path)
        assert path.exists()

    def test_manifest_is_valid_yaml(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("google/gemma-2-27b-it")
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)

    def test_manifest_model_id_correct(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("google/gemma-2-27b-it")
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        assert data["model_id"] == "google/gemma-2-27b-it"

    def test_manifest_gated_field(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info(gated=True)
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        assert data["gated"] is True

    def test_manifest_size_bytes_field(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info(size_bytes=54_000_000_000)
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        assert data["size_bytes"] == 54_000_000_000

    def test_manifest_file_count_field(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info(files=["config.json", "model.safetensors"])
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        assert data["file_count"] == 2

    def test_two_models_separate_manifests(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info1 = _make_model_info("google/gemma-2-27b-it")
        info2 = _make_model_info("meta-llama/Llama-3-8b")
        write_model_manifest(info1, tmp_path)
        write_model_manifest(info2, tmp_path)

        p1 = tmp_path / "models" / "models--google--gemma-2-27b-it" / "manifest.yaml"
        p2 = tmp_path / "models" / "models--meta-llama--Llama-3-8b" / "manifest.yaml"
        assert p1.exists()
        assert p2.exists()
        assert yaml.safe_load(p1.read_text())["model_id"] == "google/gemma-2-27b-it"
        assert yaml.safe_load(p2.read_text())["model_id"] == "meta-llama/Llama-3-8b"

    def test_slash_in_model_id_sanitized_to_double_dash(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("org/model-name")
        path = write_model_manifest(info, tmp_path)
        assert "org--model-name" in str(path)
        assert "/" not in path.parent.name

    def test_models_dir_created_if_missing(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info("google/gemma-2-27b-it")
        write_model_manifest(info, tmp_path)
        assert (tmp_path / "models").is_dir()

    def test_all_required_fields_present(self, tmp_path):
        from hf_offline.models import write_model_manifest
        info = _make_model_info()
        path = write_model_manifest(info, tmp_path)
        data = yaml.safe_load(path.read_text())
        for field in ("model_id", "pipeline_tag", "library_name", "license",
                      "gated", "tags", "file_count", "size_bytes"):
            assert field in data, f"Missing field: {field}"


# ===========================================================================
# download_model (mocked snapshot_download)
# ===========================================================================

class TestDownloadModel:
    def _setup_mocks(self, tmp_path, model_id="google/gemma-2-27b-it", gated=True):
        """Set up patches for a successful download."""
        model_dir = tmp_path / "models" / ("models--" + model_id.replace("/", "--"))
        model_dir.mkdir(parents=True)
        # Create fake model files
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        raw = _fake_raw_model(gated="auto" if gated else False)
        card = _fake_card()
        return model_dir, raw, card

    def test_gated_without_token_raises(self, tmp_path):
        raw = _fake_raw_model(gated="auto")
        card = _fake_card()
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard, \
             patch("hf_offline.models.snapshot_download"):
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            from hf_offline.models import download_model
            with pytest.raises(SystemExit):
                download_model("google/gemma-2-27b-it", tmp_path, token=None)

    def test_successful_download_returns_path_and_info(self, tmp_path):
        model_dir, raw, card = self._setup_mocks(tmp_path, gated=False)
        raw.gated = False
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard, \
             patch("hf_offline.models.snapshot_download") as mock_snap:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            mock_snap.return_value = str(model_dir)
            from hf_offline.models import download_model
            path, info = download_model("google/gemma-2-27b-it", tmp_path, token=None)
        assert isinstance(path, Path)
        assert info.model_id == "google/gemma-2-27b-it"

    def test_download_with_token_passes_token_to_snapshot(self, tmp_path):
        model_dir, raw, card = self._setup_mocks(tmp_path)
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard, \
             patch("hf_offline.models.snapshot_download") as mock_snap:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            mock_snap.return_value = str(model_dir)
            from hf_offline.models import download_model
            download_model("google/gemma-2-27b-it", tmp_path, token=HF_TOKEN)
        call_kwargs = mock_snap.call_args[1]
        assert call_kwargs["token"] == HF_TOKEN

    def test_files_cataloged_after_download(self, tmp_path):
        model_dir, raw, card = self._setup_mocks(tmp_path, gated=False)
        raw.gated = False
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard, \
             patch("hf_offline.models.snapshot_download") as mock_snap:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            mock_snap.return_value = str(model_dir)
            from hf_offline.models import download_model
            _, info = download_model("google/gemma-2-27b-it", tmp_path, token=None)
        assert len(info.files) == 2
        assert any("config.json" in f for f in info.files)

    def test_size_bytes_computed_after_download(self, tmp_path):
        model_dir, raw, card = self._setup_mocks(tmp_path, gated=False)
        raw.gated = False
        with patch("hf_offline.models.HfApi") as MockApi, \
             patch("hf_offline.models.ModelCard") as MockCard, \
             patch("hf_offline.models.snapshot_download") as mock_snap:
            MockApi.return_value.model_info.return_value = raw
            MockCard.load.return_value = card
            mock_snap.return_value = str(model_dir)
            from hf_offline.models import download_model
            _, info = download_model("google/gemma-2-27b-it", tmp_path, token=None)
        assert info.size_bytes > 0


class TestMaterializeSnapshots:
    def test_symlinks_become_files_and_blobs_removed(self, tmp_path):
        from hf_offline.models import _materialize_snapshots

        d = tmp_path / "models--a--b"
        (d / "blobs").mkdir(parents=True)
        (d / "snapshots" / "rev").mkdir(parents=True)
        (d / "blobs" / "h1").write_text("w")
        (d / "snapshots" / "rev" / "w.bin").symlink_to(d / "blobs" / "h1")
        _materialize_snapshots(d)
        f = d / "snapshots" / "rev" / "w.bin"
        assert f.is_file() and not f.is_symlink() and f.read_text() == "w"
        assert not (d / "blobs").exists()


def test_cli_env_flows_into_values_yaml(tmp_path):
    import yaml
    from hf_offline.helm import generate_chart
    from hf_offline.space import SpaceInfo

    out = generate_chart(SpaceInfo(space_id="o/s", owner="o", name="s", sdk="gradio"), tmp_path, env={"A": "1"})
    assert yaml.safe_load((out / "values.yaml").read_text())["env"] == {"A": "1"}
