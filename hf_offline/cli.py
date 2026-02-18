"""CLI interface for hf-offline."""

from __future__ import annotations

import os

import click
from rich.console import Console

from hf_offline import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="hf-offline")
def main() -> None:
    """Download HuggingFace Spaces for offline deployment on air-gapped K8s GPU clusters."""


@main.command()
@click.argument("space_id")
@click.option(
    "--model", "-m",
    multiple=True,
    help="Model ID(s) to download (e.g., meta-llama/Llama-2-7b-chat-hf). "
         "Can be specified multiple times. If omitted, auto-detected from space code.",
)
@click.option(
    "--hf-token", "-t",
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace API token (required for gated models). "
         "Also reads from HF_TOKEN env var.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output directory for the bundle. Defaults to ./bundle-<space-name>",
)
@click.option(
    "--s3-endpoint",
    default="",
    help="S3-compatible endpoint URL for model storage (e.g., http://minio:9000).",
)
@click.option(
    "--s3-bucket",
    default="models",
    help="S3 bucket name for model storage.",
)
@click.option(
    "--s3-prefix",
    default="",
    help="S3 key prefix within the bucket.",
)
@click.option(
    "--gpu-count",
    default=1,
    type=int,
    help="Number of GPUs to request in the Helm chart.",
)
@click.option(
    "--gpu-type",
    default="nvidia.com/gpu",
    help="GPU resource type for Kubernetes (e.g., nvidia.com/gpu, amd.com/gpu).",
)
@click.option(
    "--storage-size",
    default="50Gi",
    help="PVC size for model storage.",
)
@click.option(
    "--image-registry",
    default="localhost:5000",
    help="Container image registry URL for the Helm chart.",
)
@click.option(
    "--space-image",
    default=None,
    help="Pre-built container image ref for the space (to save with skopeo).",
)
@click.option(
    "--extra-image",
    multiple=True,
    help="Additional container images to export. Can be specified multiple times.",
)
@click.option(
    "--skip-images",
    is_flag=True,
    default=False,
    help="Skip exporting container images (useful if skopeo isn't available).",
)
def download(
    space_id: str,
    model: tuple[str, ...],
    hf_token: str | None,
    output: str | None,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    gpu_count: int,
    gpu_type: str,
    storage_size: str,
    image_registry: str,
    space_image: str | None,
    extra_image: tuple[str, ...],
    skip_images: bool,
) -> None:
    """Download a HuggingFace Space and all dependencies for offline deployment.

    SPACE_ID is the HuggingFace Space identifier (e.g., username/space-name).

    \b
    Examples:
      # Basic download
      hf-offline download gradio/chatbot

      # With a gated model
      hf-offline download TheBloke/Llama-2-7b-Chat-GPTQ \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --hf-token hf_xxxxx

      # Full production setup
      hf-offline download myorg/my-space \\
        --model myorg/my-model \\
        --hf-token hf_xxxxx \\
        --s3-endpoint http://minio.storage:9000 \\
        --s3-bucket models \\
        --gpu-count 2 \\
        --storage-size 100Gi \\
        --image-registry registry.internal:5000
    """
    from hf_offline.bundle import create_bundle

    model_ids = list(model) if model else None

    try:
        bundle_path = create_bundle(
            space_id,
            model_ids=model_ids,
            hf_token=hf_token,
            output_dir=output,
            s3_endpoint=s3_endpoint,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            storage_size=storage_size,
            image_registry=image_registry,
            skip_images=skip_images,
            extra_images=list(extra_image),
            space_image=space_image,
        )
        console.print(f"\n[bold green]Done![/] Bundle at: {bundle_path}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("bundle_dir")
def inspect(bundle_dir: str) -> None:
    """Inspect an existing offline bundle and show its contents."""
    import json
    from pathlib import Path

    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"

    if not manifest_path.exists():
        console.print(f"[red]No manifest.json found in {bundle_dir}[/]")
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    console.print(f"\n[bold]Bundle:[/] {bundle_dir}")
    console.print(f"  Created: {manifest.get('created', 'unknown')}")

    space = manifest.get("space", {})
    console.print(f"\n[bold]Space:[/] {space.get('id')} ({space.get('sdk')})")

    models = manifest.get("models", [])
    if models:
        console.print(f"\n[bold]Models ({len(models)}):[/]")
        for m in models:
            gated = " [yellow](gated)[/]" if m.get("gated") else ""
            console.print(f"  - {m['id']}{gated}")

    images = manifest.get("images", [])
    if images:
        console.print(f"\n[bold]Images ({len(images)}):[/]")
        for img in images:
            console.print(f"  - {img['file']}")

    steps = manifest.get("deploy_steps", [])
    if steps:
        console.print(f"\n[bold]Deploy steps:[/]")
        for step in steps:
            console.print(f"  {step}")

    console.print()
