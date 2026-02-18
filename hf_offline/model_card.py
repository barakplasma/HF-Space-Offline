"""Model card parsing and HTML rendering."""

from __future__ import annotations

from pathlib import Path

import markdown
import yaml
from rich.console import Console

console = Console()


def render_model_card_html(
    model_id: str,
    card_text: str,
    card_meta: dict,
    space_id: str = "",
    output_path: Path | None = None,
) -> str:
    """Render a model card to a self-contained HTML page.

    Args:
        model_id: The HuggingFace model ID
        card_text: Raw markdown text of the model card (without frontmatter)
        card_meta: Parsed YAML metadata from the card
        space_id: Optional associated space ID
        output_path: If provided, write the HTML to this file

    Returns:
        The rendered HTML string
    """
    # Render markdown body to HTML
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "meta"],
    )
    body_html = md.convert(card_text)

    # Build metadata badges
    badges = _build_badges(card_meta)

    # Build optional space reference line
    space_ref_html = (
        f'<div class="space-ref">Bundled with Space: {_esc(space_id)}</div>'
        if space_id
        else ""
    )

    # Build the full HTML page
    html = _HTML_TEMPLATE.format(
        model_id=_esc(model_id),
        title=_esc(card_meta.get("title", model_id)),
        space_ref=space_ref_html,
        badges=badges,
        pipeline_tag=_esc(card_meta.get("pipeline_tag", "")),
        license=_esc(card_meta.get("license", "unknown")),
        library=_esc(card_meta.get("library_name", "")),
        tags=", ".join(card_meta.get("tags", [])),
        body=body_html,
        meta_yaml=_esc(yaml.dump(card_meta, default_flow_style=False, sort_keys=False)),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        console.print(f"[green]Model card HTML written to:[/] {output_path}")

    return html


def _build_badges(meta: dict) -> str:
    """Build HTML badge elements from metadata."""
    badges = []
    if meta.get("pipeline_tag"):
        badges.append(f'<span class="badge pipeline">{_esc(meta["pipeline_tag"])}</span>')
    if meta.get("license"):
        badges.append(f'<span class="badge license">{_esc(meta["license"])}</span>')
    if meta.get("library_name"):
        badges.append(f'<span class="badge library">{_esc(meta["library_name"])}</span>')
    for tag in meta.get("tags", [])[:8]:
        badges.append(f'<span class="badge tag">{_esc(tag)}</span>')
    return "\n      ".join(badges)


def _esc(s: str) -> str:
    """Basic HTML escaping."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Model Card</title>
  <style>
    :root {{
      --bg: #0d1117;
      --fg: #c9d1d9;
      --muted: #8b949e;
      --accent: #58a6ff;
      --surface: #161b22;
      --border: #30363d;
      --green: #3fb950;
      --orange: #d29922;
      --purple: #bc8cff;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.6;
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }}
    h1 {{
      font-size: 1.8rem;
      color: #f0f6fc;
      margin-bottom: 0.5rem;
    }}
    .model-id {{
      color: var(--accent);
      font-family: monospace;
      font-size: 0.9rem;
    }}
    .space-ref {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 0.3rem;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 1rem;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge.pipeline {{ background: #1f3a5f; color: var(--accent); }}
    .badge.license {{ background: #2a1f0a; color: var(--orange); }}
    .badge.library {{ background: #1a0f30; color: var(--purple); }}
    .badge.tag {{ background: var(--surface); color: var(--muted); border: 1px solid var(--border); }}

    .content {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2rem;
    }}
    .content h1, .content h2, .content h3, .content h4 {{
      color: #f0f6fc;
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.3rem;
    }}
    .content h1:first-child, .content h2:first-child {{ margin-top: 0; }}
    .content p {{ margin: 0.8rem 0; }}
    .content code {{
      background: #1a2332;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    .content pre {{
      background: #1a2332;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      margin: 1rem 0;
    }}
    .content pre code {{ background: none; padding: 0; }}
    .content a {{ color: var(--accent); }}
    .content table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
    }}
    .content th, .content td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.75rem;
      text-align: left;
    }}
    .content th {{ background: var(--bg); }}
    .content ul, .content ol {{
      margin: 0.5rem 0;
      padding-left: 1.5rem;
    }}
    .content li {{ margin: 0.25rem 0; }}
    .content blockquote {{
      border-left: 3px solid var(--accent);
      padding-left: 1rem;
      color: var(--muted);
      margin: 1rem 0;
    }}

    details {{
      margin-top: 2rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem;
    }}
    details summary {{
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
    }}
    details pre {{
      margin-top: 0.5rem;
      font-size: 0.8rem;
      color: var(--muted);
      white-space: pre-wrap;
    }}

    footer {{
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.8rem;
      text-align: center;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="model-id">{model_id}</div>
    {space_ref}
    <div class="badges">
      {badges}
    </div>
  </header>

  <div class="content">
    {body}
  </div>

  <details>
    <summary>Raw Metadata (YAML)</summary>
    <pre>{meta_yaml}</pre>
  </details>

  <footer>
    Generated by hf-offline &mdash; HuggingFace Space Offline Bundler
  </footer>
</body>
</html>
"""
