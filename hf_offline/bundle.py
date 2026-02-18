"""Bundle orchestrator - ties all components together into a single output folder."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

from hf_offline.helm import generate_chart
from hf_offline.images import export_space_images
from hf_offline.model_card import render_model_card_html
from hf_offline.models import ModelInfo, download_model, write_model_manifest
from hf_offline.space import SpaceInfo, clone_space, generate_dockerfile

console = Console()


def create_bundle(
    space_id: str,
    *,
    model_ids: list[str] | None = None,
    hf_token: str | None = None,
    output_dir: str | None = None,
    s3_endpoint: str = "",
    s3_bucket: str = "models",
    s3_prefix: str = "",
    gpu_count: int = 1,
    gpu_type: str = "nvidia.com/gpu",
    storage_size: str = "50Gi",
    image_registry: str = "localhost:5000",
    skip_images: bool = False,
    extra_images: list[str] | None = None,
    space_image: str | None = None,
) -> Path:
    """Create a complete offline bundle for a HuggingFace Space.

    The bundle is a single self-contained folder (Pinokio-inspired) with:
    - app/          Space application code
    - models/       Model weights (separate, for PVC mount + S3 upload)
    - images/       OCI container image tarballs (via skopeo)
    - chart/        Helm chart for K8s deployment
    - model-card.html  Rendered model card
    - manifest.json    Bundle metadata
    """
    # Determine output directory
    safe_name = space_id.replace("/", "--")
    dest = Path(output_dir) if output_dir else Path.cwd() / f"bundle-{safe_name}"

    if dest.exists():
        console.print(f"[yellow]Output directory exists, will merge into:[/] {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Bundling HuggingFace Space for offline deployment[/]\n"
        f"Space: [cyan]{space_id}[/]\n"
        f"Output: [cyan]{dest}[/]",
        title="hf-offline",
    ))

    # -----------------------------------------------------------------------
    # Step 1: Download the Space
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Step 1: Download Space")
    space_dir, space_info = clone_space(space_id, dest, token=hf_token)

    # If user didn't specify model IDs, use what we detected
    if not model_ids:
        model_ids = space_info.models

    if not model_ids:
        console.print("[yellow]No models detected. Specify --model to download models.[/]")

    # -----------------------------------------------------------------------
    # Step 2: Generate Dockerfile (if needed)
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Step 2: Generate Dockerfile")
    generate_dockerfile(space_info, dest / "app")

    # -----------------------------------------------------------------------
    # Step 3: Download models
    # -----------------------------------------------------------------------
    model_infos: list[ModelInfo] = []
    if model_ids:
        console.rule("[bold blue]Step 3: Download Models")
        for mid in model_ids:
            console.print(f"\n[bold]Model:[/] {mid}")
            try:
                _, minfo = download_model(mid, dest, token=hf_token)
                model_infos.append(minfo)
                write_model_manifest(minfo, dest)
            except Exception as e:
                console.print(f"[red]Failed to download model {mid}:[/] {e}")
                console.print("[yellow]Continuing with remaining models...[/]")
    else:
        console.rule("[bold blue]Step 3: Download Models (skipped - none specified)")

    # -----------------------------------------------------------------------
    # Step 4: Export container images
    # -----------------------------------------------------------------------
    saved_images: list[Path] = []
    if not skip_images:
        console.rule("[bold blue]Step 4: Export Container Images")
        saved_images = export_space_images(
            sdk=space_info.sdk,
            dest=dest,
            space_image=space_image,
            extra_images=extra_images,
        )
        if not saved_images:
            console.print(
                "[yellow]No images saved. Install skopeo to export images.[/]"
            )
    else:
        console.rule("[bold blue]Step 4: Export Container Images (skipped)")

    # -----------------------------------------------------------------------
    # Step 5: Generate Helm chart
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Step 5: Generate Helm Chart")
    chart_dir = generate_chart(
        space_info,
        dest,
        model_ids=model_ids,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        storage_size=storage_size,
        image_registry=image_registry,
    )

    # -----------------------------------------------------------------------
    # Step 6: Render model card HTML
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Step 6: Render Model Card")
    for minfo in model_infos:
        if minfo.card_text:
            render_model_card_html(
                model_id=minfo.model_id,
                card_text=minfo.card_text,
                card_meta=minfo.card_meta,
                space_id=space_id,
                output_path=dest / "model-card.html",
            )
            # Also copy into chart for optional serving
            model_card_in_chart = chart_dir / "model-card.html"
            shutil.copy2(dest / "model-card.html", model_card_in_chart)
            break  # Use first model's card as primary
    else:
        console.print("[yellow]No model card content to render.[/]")

    # -----------------------------------------------------------------------
    # Step 7: Write bundle manifest
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Step 7: Write Bundle Manifest")
    manifest = _build_manifest(space_info, model_infos, saved_images, dest)
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"[green]Manifest written:[/] {manifest_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_summary(dest, space_info, model_infos, saved_images)

    return dest


def _build_manifest(
    space_info: SpaceInfo,
    model_infos: list[ModelInfo],
    saved_images: list[Path],
    dest: Path,
) -> dict:
    """Build the bundle manifest (inspired by Pinokio's pinokio.js)."""
    return {
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": "hf-offline",
        "space": {
            "id": space_info.space_id,
            "sdk": space_info.sdk,
            "title": space_info.title,
            "port": space_info.port,
        },
        "models": [
            {
                "id": m.model_id,
                "pipeline_tag": m.pipeline_tag,
                "gated": m.gated,
                "size_bytes": m.size_bytes,
                "path": f"models/{m.model_id.replace('/', '--')}",
            }
            for m in model_infos
        ],
        "images": [
            {"file": img.name, "path": f"images/{img.name}"}
            for img in saved_images
        ],
        "chart": "chart/",
        "contents": {
            "app": "app/",
            "models": "models/",
            "images": "images/",
            "chart": "chart/",
            "model_card": "model-card.html",
        },
        "deploy_steps": [
            "1. Load container images into your registry: cd images && ./load-images.sh <registry-url>",
            "2. Upload models to S3: mc cp --recursive models/ s3store/<bucket>/",
            "3. Build the space image: cd app && docker build -t <registry>/<name>:latest .",
            "4. Update chart/values.yaml with your registry, S3, and GPU settings",
            "5. Deploy: helm install <release-name> chart/",
        ],
    }


def _print_summary(
    dest: Path,
    space_info: SpaceInfo,
    model_infos: list[ModelInfo],
    saved_images: list[Path],
) -> None:
    """Print a summary of what was bundled."""
    total_model_size = sum(m.size_bytes for m in model_infos)
    size_str = _human_size(total_model_size)

    lines = [
        f"[bold green]Bundle complete![/]\n",
        f"  Space:      [cyan]{space_info.space_id}[/] ({space_info.sdk})",
        f"  Models:     {len(model_infos)} ({size_str})",
        f"  Images:     {len(saved_images)} tarballs",
        f"  Location:   [cyan]{dest}[/]",
        "",
        "[bold]Bundle contents:[/]",
        f"  app/          Space application code + Dockerfile",
        f"  models/       Model weights (mount as PVC / upload to S3)",
        f"  images/       Container image tarballs (load with skopeo)",
        f"  chart/        Helm chart for Kubernetes deployment",
        f"  model-card.html  Rendered model card",
        f"  manifest.json    Bundle metadata",
        "",
        "[bold]Next steps:[/]",
        "  1. Transfer this folder to your air-gapped environment",
        "  2. Load images:  cd images && ./load-images.sh <registry>",
        "  3. Upload models: upload models/ to your S3 bucket",
        "  4. Build app:    cd app && docker build -t <registry>/<name> .",
        "  5. Edit:         chart/values.yaml",
        "  6. Deploy:       helm install <name> chart/",
    ]

    console.print(Panel("\n".join(lines), title="Summary"))


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
