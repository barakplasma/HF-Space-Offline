"""HuggingFace Space metadata fetching, cloning, and Dockerfile generation."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from rich.console import Console

console = Console()

SDK_PORTS = {
    "gradio": 7860,
    "streamlit": 8501,
    "docker": 8080,
    "static": 80,
}

SDK_BASE_IMAGES = {
    "gradio": "python:3.10-slim",
    "streamlit": "python:3.10-slim",
}


@dataclass
class SpaceInfo:
    """Parsed metadata about a HuggingFace Space."""

    space_id: str
    owner: str
    name: str
    sdk: str
    sdk_version: str | None = None
    python_version: str | None = None
    app_file: str | None = None
    port: int = 7860
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    title: str = ""
    emoji: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    has_dockerfile: bool = False
    has_requirements: bool = False
    pinned: bool = False
    # Raw metadata dict for anything we didn't parse
    raw_meta: dict = field(default_factory=dict)


def fetch_space_metadata(space_id: str, token: str | None = None) -> SpaceInfo:
    """Fetch space metadata from the HuggingFace Hub API."""
    api = HfApi(token=token)
    space_info_raw = api.space_info(repo_id=space_id)

    owner, name = space_id.split("/", 1)

    # Parse the README frontmatter for richer metadata
    raw_meta = {}
    try:
        readme_content = api.hf_hub_download(
            repo_id=space_id, filename="README.md", repo_type="space"
        )
        readme_path = Path(readme_content)
        text = readme_path.read_text(encoding="utf-8")
        raw_meta = _parse_yaml_frontmatter(text)
    except Exception:
        pass

    sdk = getattr(space_info_raw, "sdk", None) or raw_meta.get("sdk", "gradio")
    sdk = sdk.lower()

    # Detect models referenced in metadata
    models = raw_meta.get("models", [])
    if isinstance(models, str):
        models = [models]

    info = SpaceInfo(
        space_id=space_id,
        owner=owner,
        name=name,
        sdk=sdk,
        sdk_version=raw_meta.get("sdk_version"),
        python_version=raw_meta.get("python_version", "3.10"),
        app_file=raw_meta.get("app_file", "app.py"),
        port=SDK_PORTS.get(sdk, 7860),
        models=models,
        datasets=raw_meta.get("datasets", []) if isinstance(raw_meta.get("datasets"), list) else [],
        title=raw_meta.get("title", name),
        emoji=raw_meta.get("emoji", ""),
        license=raw_meta.get("license", ""),
        tags=raw_meta.get("tags", []),
        raw_meta=raw_meta,
    )
    return info


def clone_space(
    space_id: str,
    dest: Path,
    token: str | None = None,
) -> tuple[Path, SpaceInfo]:
    """Clone a HuggingFace Space repository and return its path + metadata."""
    console.print(f"[bold blue]Downloading space:[/] {space_id}")

    space_dir = dest / "app"
    if space_dir.exists():
        shutil.rmtree(space_dir)

    # Download the full space repo
    downloaded = snapshot_download(
        repo_id=space_id,
        repo_type="space",
        local_dir=str(space_dir),
        token=token,
    )
    console.print(f"[green]Space downloaded to:[/] {downloaded}")

    info = fetch_space_metadata(space_id, token)

    # Check for Dockerfile and requirements.txt
    info.has_dockerfile = (space_dir / "Dockerfile").exists()
    info.has_requirements = (space_dir / "requirements.txt").exists()

    # Scan code for additional model references
    code_models = scan_for_model_refs(space_dir)
    for m in code_models:
        if m not in info.models:
            info.models.append(m)

    return Path(downloaded), info


def scan_for_model_refs(space_dir: Path) -> list[str]:
    """Scan Python files in the space for model ID references."""
    model_patterns = [
        # from_pretrained("org/model")
        r'from_pretrained\s*\(\s*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']',
        # pipeline("task", model="org/model")
        r'model\s*=\s*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']',
        # HfApi().model_info("org/model")
        r'repo_id\s*=\s*["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']',
    ]

    found = set()
    for py_file in space_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in model_patterns:
                matches = re.findall(pattern, text)
                found.update(matches)
        except Exception:
            continue

    # Filter out common false positives
    false_positives = {
        "gradio/gradio", "streamlit/streamlit", "huggingface/hub",
        "numpy/numpy", "pandas/pandas",
    }
    return [m for m in sorted(found) if m not in false_positives]


def generate_dockerfile(info: SpaceInfo, space_dir: Path) -> Path:
    """Generate a Dockerfile for the space if one doesn't exist."""
    dockerfile_path = space_dir / "Dockerfile"

    if info.has_dockerfile:
        console.print("[yellow]Space already has a Dockerfile, keeping it.[/]")
        return dockerfile_path

    if info.sdk == "gradio":
        content = _gradio_dockerfile(info)
    elif info.sdk == "streamlit":
        content = _streamlit_dockerfile(info)
    else:
        content = _generic_dockerfile(info)

    dockerfile_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Generated Dockerfile for {info.sdk} space[/]")
    return dockerfile_path


def _gradio_dockerfile(info: SpaceInfo) -> str:
    python_ver = info.python_version or "3.10"
    app_file = info.app_file or "app.py"
    reqs = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt" if info.has_requirements else "RUN pip install --no-cache-dir gradio"

    return f"""\
FROM python:{python_ver}-slim

WORKDIR /app

{reqs}

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT={info.port}

EXPOSE {info.port}

CMD ["python", "{app_file}"]
"""


def _streamlit_dockerfile(info: SpaceInfo) -> str:
    python_ver = info.python_version or "3.10"
    app_file = info.app_file or "app.py"
    reqs = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt" if info.has_requirements else "RUN pip install --no-cache-dir streamlit"

    return f"""\
FROM python:{python_ver}-slim

WORKDIR /app

{reqs}

COPY . .

EXPOSE {info.port}

CMD ["streamlit", "run", "{app_file}", "--server.address=0.0.0.0", "--server.port={info.port}"]
"""


def _generic_dockerfile(info: SpaceInfo) -> str:
    python_ver = info.python_version or "3.10"
    app_file = info.app_file or "app.py"

    return f"""\
FROM python:{python_ver}-slim

WORKDIR /app

COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

EXPOSE {info.port}

CMD ["python", "{app_file}"]
"""


def _parse_yaml_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from a markdown file."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}
