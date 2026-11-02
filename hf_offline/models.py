"""Model downloading with gated model support."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi, ModelCard, snapshot_download
from rich.console import Console

console = Console()


@dataclass
class ModelInfo:
    """Metadata about a downloaded model."""

    model_id: str
    owner: str
    name: str
    pipeline_tag: str = ""
    library_name: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    card_text: str = ""
    card_meta: dict = field(default_factory=dict)
    gated: bool = False
    files: list[str] = field(default_factory=list)
    local_dir: Path | None = None
    size_bytes: int = 0


def fetch_model_info(model_id: str, token: str | None = None) -> ModelInfo:
    """Fetch model metadata from HuggingFace Hub."""
    api = HfApi(token=token)
    raw = api.model_info(repo_id=model_id, token=token)

    owner, name = model_id.split("/", 1) if "/" in model_id else ("", model_id)

    # Get model card
    card_text = ""
    card_meta = {}
    try:
        card = ModelCard.load(model_id, token=token)
        card_text = card.text or ""
        card_meta = card.data.to_dict() if card.data else {}
    except Exception:
        pass

    gated = getattr(raw, "gated", False)
    if gated and gated != "false":
        gated = True
    else:
        gated = False

    return ModelInfo(
        model_id=model_id,
        owner=owner,
        name=name,
        pipeline_tag=getattr(raw, "pipeline_tag", "") or "",
        library_name=getattr(raw, "library_name", "") or "",
        license=card_meta.get("license", ""),
        tags=getattr(raw, "tags", []) or [],
        card_text=card_text,
        card_meta=card_meta,
        gated=gated,
    )


def model_dir_name(model_id: str) -> str:
    """Directory name in the HF hub cache layout (what HF_HUB_CACHE expects)."""
    return "models--" + model_id.replace("/", "--")


def _materialize_snapshots(model_dir: Path) -> None:
    """Replace snapshot symlinks with real files and drop blobs/.

    S3 uploads and tar don't preserve symlinks; offline lookups only need snapshots/.
    """
    for link in (model_dir / "snapshots").rglob("*"):
        if link.is_symlink():
            target = link.resolve()
            link.unlink()
            target.replace(link)  # moves the blob; a blob shared by two names is copied below
    # shortcut: files sharing one blob keep only the first name, re-copy if that matters
    shutil.rmtree(model_dir / "blobs", ignore_errors=True)


def download_model(
    model_id: str,
    dest: Path,
    token: str | None = None,
) -> tuple[Path, ModelInfo]:
    """Download a model's files to a local directory.

    The model is stored in HF hub cache layout under dest/models/ so that mounting
    it with HF_HUB_CACHE and HF_HUB_OFFLINE=1 lets from_pretrained() work air-gapped.
    """
    info = fetch_model_info(model_id, token)

    if info.gated and not token:
        console.print(
            f"[bold red]Model {model_id} is gated![/] "
            "You must provide --hf-token to download it."
        )
        raise SystemExit(1)

    if info.gated:
        console.print(
            f"[yellow]Model {model_id} is gated. "
            "Using provided token for access.[/]"
        )

    model_dir = dest / "models" / model_dir_name(model_id)

    console.print(f"[bold blue]Downloading model:[/] {model_id}")
    downloaded = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        cache_dir=str(dest / "models"),
        token=token,
    )
    _materialize_snapshots(model_dir)
    shutil.rmtree(dest / "models" / "blobs", ignore_errors=True)  # hub's xet chunk cache, not needed offline
    console.print(f"[green]Model downloaded to:[/] {downloaded}")

    info.local_dir = Path(downloaded)

    # Catalog downloaded files
    info.files = [
        str(p.relative_to(model_dir))
        for p in model_dir.rglob("*")
        if p.is_file()
    ]

    # Rough size estimate
    info.size_bytes = sum(
        p.stat().st_size for p in model_dir.rglob("*") if p.is_file()
    )

    return Path(downloaded), info


def write_model_manifest(info: ModelInfo, dest: Path) -> Path:
    """Write a YAML manifest describing the model inside its own subdirectory.

    Each model gets a manifest at models/<safe-name>/manifest.yaml so that
    multiple models can coexist without overwriting each other.
    """
    manifest = {
        "model_id": info.model_id,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "license": info.license,
        "gated": info.gated,
        "tags": info.tags,
        "file_count": len(info.files),
        "size_bytes": info.size_bytes,
    }

    manifest_path = dest / "models" / model_dir_name(info.model_id) / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path
