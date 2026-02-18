"""Comprehensive tests for hf_offline/images.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# check_skopeo
# ===========================================================================

class TestCheckSkopeo:
    def test_returns_true_when_found(self):
        with patch("hf_offline.images.shutil.which", return_value="/usr/bin/skopeo"):
            from hf_offline.images import check_skopeo
            assert check_skopeo() is True

    def test_returns_false_when_not_found(self):
        with patch("hf_offline.images.shutil.which", return_value=None):
            from hf_offline.images import check_skopeo
            assert check_skopeo() is False


# ===========================================================================
# _safe_filename
# ===========================================================================

class TestSafeFilename:
    def _safe(self, ref):
        from hf_offline.images import _safe_filename
        return _safe_filename(ref)

    def test_slashes_replaced(self):
        assert "/" not in self._safe("org/image:tag")

    def test_colons_replaced(self):
        assert ":" not in self._safe("image:tag")

    def test_dots_replaced_with_dashes(self):
        assert "." not in self._safe("python:3.10-slim")

    def test_specific_example(self):
        result = self._safe("python:3.10-slim")
        assert result == "python_3-10-slim"

    def test_minio_mc(self):
        result = self._safe("minio/mc:latest")
        assert result == "minio_mc_latest"


# ===========================================================================
# _ref_to_tag
# ===========================================================================

class TestRefToTag:
    def _ref(self, src):
        from hf_offline.images import _ref_to_tag
        return _ref_to_tag(src)

    def test_strips_docker_prefix(self):
        assert self._ref("docker://python:3.10-slim") == "python:3.10-slim"

    def test_no_prefix_unchanged(self):
        assert self._ref("python:3.10-slim") == "python:3.10-slim"

    def test_minio_ref(self):
        assert self._ref("docker://minio/mc:latest") == "minio/mc:latest"


# ===========================================================================
# skopeo_copy
# ===========================================================================

class TestSkopeoCoopy:
    def test_returns_true_on_success(self, tmp_path):
        tarball = tmp_path / "images" / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            from hf_offline.images import skopeo_copy
            result = skopeo_copy("docker://python:3.10-slim", tarball)
        assert result is True

    def test_returns_false_on_nonzero_exit(self, tmp_path):
        tarball = tmp_path / "images" / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            from hf_offline.images import skopeo_copy
            result = skopeo_copy("docker://python:3.10-slim", tarball)
        assert result is False

    def test_returns_false_on_timeout(self, tmp_path):
        tarball = tmp_path / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["skopeo"], timeout=600)
            from hf_offline.images import skopeo_copy
            result = skopeo_copy("docker://python:3.10-slim", tarball)
        assert result is False

    def test_returns_false_when_skopeo_not_found(self, tmp_path):
        tarball = tmp_path / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            from hf_offline.images import skopeo_copy
            result = skopeo_copy("docker://python:3.10-slim", tarball)
        assert result is False

    def test_parent_dir_created(self, tmp_path):
        tarball = tmp_path / "deep" / "nested" / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            from hf_offline.images import skopeo_copy
            skopeo_copy("docker://python:3.10-slim", tarball)
        assert tarball.parent.exists()

    def test_src_creds_passed_to_command(self, tmp_path):
        tarball = tmp_path / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            from hf_offline.images import skopeo_copy
            skopeo_copy("docker://private/image:tag", tarball, src_creds="user:pass")
        cmd = mock_run.call_args[0][0]
        assert "--src-creds" in cmd
        assert "user:pass" in cmd

    def test_command_includes_additional_tag(self, tmp_path):
        tarball = tmp_path / "image.tar"
        with patch("hf_offline.images.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            from hf_offline.images import skopeo_copy
            skopeo_copy("docker://python:3.10-slim", tarball)
        cmd = mock_run.call_args[0][0]
        assert "--additional-tag" in cmd


# ===========================================================================
# export_space_images
# ===========================================================================

class TestExportSpaceImages:
    def test_returns_empty_when_no_skopeo(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=False):
            from hf_offline.images import export_space_images
            result = export_space_images("gradio", tmp_path)
        assert result == []

    def test_gradio_exports_python_base_image(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("gradio", tmp_path)
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        assert any("python:3.10-slim" in s for s in srcs)

    def test_streamlit_exports_python_base_image(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("streamlit", tmp_path)
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        assert any("python:3.10-slim" in s for s in srcs)

    def test_docker_sdk_no_base_image_exported(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("docker", tmp_path)
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        # Docker SDK has no auto-generated base image
        assert not any("python:3.10-slim" in s for s in srcs)

    def test_minio_mc_always_exported_when_skopeo_present(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("gradio", tmp_path)
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        assert any("minio" in s for s in srcs)

    def test_space_image_exported_when_provided(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("docker", tmp_path, space_image="registry/my-space:v1")
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        assert any("registry/my-space:v1" in s for s in srcs)

    def test_extra_images_exported(self, tmp_path):
        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy") as mock_copy:
            mock_copy.return_value = True
            from hf_offline.images import export_space_images
            export_space_images("docker", tmp_path, extra_images=["my-registry/extra:1.0"])
        srcs = [call[0][0] for call in mock_copy.call_args_list]
        assert any("my-registry/extra:1.0" in s for s in srcs)

    def test_returns_only_successful_saves(self, tmp_path):
        call_count = [0]
        def side_effect(src, dest, *args, **kwargs):
            # Create the tarball file to simulate success
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
            call_count[0] += 1
            # Fail on the second call (minio/mc)
            return call_count[0] != 2

        with patch("hf_offline.images.check_skopeo", return_value=True), \
             patch("hf_offline.images.skopeo_copy", side_effect=side_effect):
            from hf_offline.images import export_space_images
            result = export_space_images("gradio", tmp_path)
        # Should have 1 success (python base) and 1 failure (minio), so 1 returned
        assert len(result) == 1


# ===========================================================================
# _write_load_script
# ===========================================================================

class TestWriteLoadScript:
    def test_script_created(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        tarballs = [images_dir / "image1.tar", images_dir / "image2.tar"]
        for t in tarballs:
            t.touch()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, tarballs)
        assert (images_dir / "load-images.sh").exists()

    def test_script_is_executable(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        tarballs = [images_dir / "image.tar"]
        tarballs[0].touch()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, tarballs)
        import stat
        mode = (images_dir / "load-images.sh").stat().st_mode
        assert mode & stat.S_IXUSR

    def test_script_references_all_tarballs(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        tarballs = [images_dir / "alpha.tar", images_dir / "beta.tar"]
        for t in tarballs:
            t.touch()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, tarballs)
        content = (images_dir / "load-images.sh").read_text()
        assert "alpha.tar" in content
        assert "beta.tar" in content

    def test_script_has_shebang(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        tarballs = [images_dir / "image.tar"]
        tarballs[0].touch()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, tarballs)
        content = (images_dir / "load-images.sh").read_text()
        assert content.startswith("#!/usr/bin/env bash")

    def test_empty_list_no_script_created(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, [])
        assert not (images_dir / "load-images.sh").exists()

    def test_script_uses_registry_variable(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        tarballs = [images_dir / "image.tar"]
        tarballs[0].touch()
        from hf_offline.images import _write_load_script
        _write_load_script(images_dir, tarballs)
        content = (images_dir / "load-images.sh").read_text()
        assert "$REGISTRY" in content
