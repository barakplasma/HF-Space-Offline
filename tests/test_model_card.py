"""Comprehensive tests for hf_offline/model_card.py."""

from __future__ import annotations

from pathlib import Path

import pytest


def _render(
    model_id="google/gemma-2-27b-it",
    card_text="# Gemma\n\nA translation model.",
    card_meta=None,
    space_id="hysts/translategemma-27b-it",
    output_path=None,
):
    from hf_offline.model_card import render_model_card_html
    if card_meta is None:
        card_meta = {
            "license": "gemma",
            "pipeline_tag": "text2text-generation",
            "library_name": "transformers",
            "tags": ["translation", "gemma"],
        }
    return render_model_card_html(
        model_id=model_id,
        card_text=card_text,
        card_meta=card_meta,
        space_id=space_id,
        output_path=output_path,
    )


# ===========================================================================
# Template rendering correctness
# ===========================================================================

class TestRenderModelCardHtml:
    def test_returns_string(self):
        assert isinstance(_render(), str)

    def test_is_valid_html_doctype(self):
        html = _render()
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_no_jinja2_artifacts(self):
        html = _render()
        assert "{%" not in html
        assert "%}" not in html

    def test_no_unresolved_format_placeholders(self):
        html = _render()
        import re
        # Check for leftover {word} placeholders (excluding CSS {})
        # Only flag single-brace placeholders that look like format keys
        leftover = re.findall(r'\{[a-z_]+\}', html)
        assert leftover == [], f"Leftover format placeholders: {leftover}"

    def test_model_id_in_output(self):
        html = _render(model_id="google/gemma-2-27b-it")
        assert "google/gemma-2-27b-it" in html

    def test_title_used_when_provided(self):
        html = _render(card_meta={"title": "My Custom Title"})
        assert "My Custom Title" in html

    def test_model_id_as_title_fallback(self):
        html = _render(model_id="google/gemma-2-27b-it", card_meta={})
        assert "google/gemma-2-27b-it" in html

    def test_space_ref_shown_when_provided(self):
        html = _render(space_id="hysts/translategemma-27b-it")
        assert "Bundled with Space" in html
        assert "hysts/translategemma-27b-it" in html

    def test_space_ref_absent_when_empty_string(self):
        html = _render(space_id="")
        assert "Bundled with Space" not in html

    def test_pipeline_tag_badge_present(self):
        html = _render(card_meta={"pipeline_tag": "text2text-generation"})
        assert "text2text-generation" in html

    def test_license_badge_present(self):
        html = _render(card_meta={"license": "apache-2.0"})
        assert "apache-2.0" in html

    def test_library_badge_present(self):
        html = _render(card_meta={"library_name": "transformers"})
        assert "transformers" in html

    def test_tag_badges_present(self):
        html = _render(card_meta={"tags": ["translation", "gemma"]})
        assert "translation" in html
        assert "gemma" in html

    def test_max_eight_tag_badges(self):
        tags = [f"tag-{i}" for i in range(20)]
        html = _render(card_meta={"tags": tags})
        # Only first 8 tags should appear as badges
        for i in range(8):
            assert f"tag-{i}" in html

    def test_tags_with_non_string_values(self):
        """Tags may be integers or other types — should not crash."""
        html = _render(card_meta={"tags": [1, 2, "text"]})
        assert "1" in html
        assert "2" in html

    def test_markdown_body_converted_to_html(self):
        html = _render(card_text="# Heading\n\nParagraph text.")
        assert "<h1>" in html
        assert "<p>" in html

    def test_raw_metadata_yaml_section_present(self):
        html = _render(card_meta={"license": "gemma"})
        assert "Raw Metadata" in html
        assert "license" in html

    def test_html_escaping_in_model_id(self):
        html = _render(model_id='<script>alert(1)</script>')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_escaping_in_space_id(self):
        html = _render(space_id='<b>evil</b>')
        assert "<b>evil</b>" not in html

    def test_html_escaping_in_pipeline_tag(self):
        html = _render(card_meta={"pipeline_tag": "<xss>"})
        assert "<xss>" not in html
        assert "&lt;xss&gt;" in html

    def test_html_escaping_in_license(self):
        html = _render(card_meta={"license": '<"evil">'})
        assert '<"evil">' not in html

    def test_empty_card_text(self):
        html = _render(card_text="")
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_empty_card_meta(self):
        html = _render(card_meta={})
        assert isinstance(html, str)

    def test_footer_present(self):
        html = _render()
        assert "hf-offline" in html

    def test_ampersand_in_model_id_escaped(self):
        html = _render(model_id="org/model&version")
        assert "model&version" not in html
        assert "model&amp;version" in html


# ===========================================================================
# File output
# ===========================================================================

class TestRenderToFile:
    def test_file_created(self, tmp_path):
        out = tmp_path / "model-card.html"
        _render(output_path=out)
        assert out.exists()

    def test_file_contains_html(self, tmp_path):
        out = tmp_path / "model-card.html"
        _render(output_path=out)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_parent_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "card.html"
        _render(output_path=out)
        assert out.exists()

    def test_file_is_utf8(self, tmp_path):
        out = tmp_path / "card.html"
        _render(
            card_text="# Model 🌍\n\nEmoji support.",
            output_path=out,
        )
        content = out.read_bytes().decode("utf-8")
        assert "Model" in content

    def test_returns_html_string_even_when_file_written(self, tmp_path):
        out = tmp_path / "card.html"
        result = _render(output_path=out)
        assert isinstance(result, str)
        assert result == out.read_text(encoding="utf-8")


# ===========================================================================
# _build_badges helper
# ===========================================================================

class TestBuildBadges:
    def _badges(self, meta):
        from hf_offline.model_card import _build_badges
        return _build_badges(meta)

    def test_pipeline_badge_class(self):
        badges = self._badges({"pipeline_tag": "translation"})
        assert 'class="badge pipeline"' in badges

    def test_license_badge_class(self):
        badges = self._badges({"license": "apache-2.0"})
        assert 'class="badge license"' in badges

    def test_library_badge_class(self):
        badges = self._badges({"library_name": "transformers"})
        assert 'class="badge library"' in badges

    def test_tag_badge_class(self):
        badges = self._badges({"tags": ["my-tag"]})
        assert 'class="badge tag"' in badges

    def test_empty_meta_produces_no_badges(self):
        badges = self._badges({})
        assert badges.strip() == ""

    def test_tag_limit_to_eight(self):
        meta = {"tags": [f"t{i}" for i in range(20)]}
        badges = self._badges(meta)
        assert badges.count('class="badge tag"') == 8


# ===========================================================================
# _esc helper
# ===========================================================================

class TestEsc:
    def _esc(self, s):
        from hf_offline.model_card import _esc
        return _esc(s)

    def test_ampersand(self):
        assert self._esc("a & b") == "a &amp; b"

    def test_lt(self):
        assert self._esc("<tag>") == "&lt;tag&gt;"

    def test_double_quote(self):
        assert self._esc('"hello"') == "&quot;hello&quot;"

    def test_safe_string_unchanged(self):
        assert self._esc("hello world") == "hello world"

    def test_integer_converted_to_string(self):
        assert self._esc(42) == "42"

    def test_none_converted_to_string(self):
        assert self._esc(None) == "None"

    def test_xss_attack_escaped(self):
        result = self._esc('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
