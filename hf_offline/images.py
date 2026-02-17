"""Container image export via skopeo for air-gapped deployment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# Known base images needed for Gradio/Streamlit spaces
RUNTIME_IMAGES = {
    "gradio": "python:3.10-slim",
    "streamlit": "python:3.10-slim",
}

# S3 sync init container image
S3_SYNC_IMAGE = "minio/mc:latest"


def check_skopeo() -> bool:
    """Check if skopeo is available on the system."""
    return shutil.which("skopeo") is not None


def skopeo_copy(
    src: str,
    dest_path: Path,
    src_creds: str | None = None,
) -> bool:
    """Copy a container image using skopeo to an OCI archive.

    Args:
        src: Source image reference (e.g., "docker://python:3.10-slim")
        dest_path: Local path for the OCI archive tarball
        src_creds: Optional "user:password" for source registry auth
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "skopeo", "copy",
        "--additional-tag", _ref_to_tag(src),
    ]

    if src_creds:
        cmd.extend(["--src-creds", src_creds])

    cmd.extend([src, f"oci-archive:{dest_path}"])

    console.print(f"[bold blue]Saving image:[/] {src}")
    console.print(f"  -> {dest_path}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            console.print(f"[red]skopeo error:[/] {result.stderr.strip()}")
            return False
        console.print(f"[green]Image saved:[/] {dest_path.name}")
        return True
    except subprocess.TimeoutExpired:
        console.print("[red]skopeo timed out after 10 minutes[/]")
        return False
    except FileNotFoundError:
        console.print("[red]skopeo not found. Install it: https://github.com/containers/skopeo/blob/main/install.md[/]")
        return False


def export_space_images(
    sdk: str,
    dest: Path,
    space_image: str | None = None,
    extra_images: list[str] | None = None,
    registry_creds: str | None = None,
) -> list[Path]:
    """Export all container images needed for a space deployment.

    Exports:
    1. The space's own container image (if built/available)
    2. The base runtime image for the SDK
    3. The S3 sync init container image (minio/mc)
    4. Any extra images specified

    Returns list of saved tarball paths.
    """
    if not check_skopeo():
        console.print(
            "[bold red]skopeo is not installed![/]\n"
            "Install it to export container images for air-gapped deployment.\n"
            "See: https://github.com/containers/skopeo/blob/main/install.md\n"
            "\n"
            "Alternatively, you can use:\n"
            "  - Zarf: https://docs.zarf.dev/\n"
            "  - Hauler: https://hauler.dev/\n"
        )
        return []

    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Space image (if pre-built and available)
    if space_image:
        tarball = images_dir / "space-app.tar"
        if skopeo_copy(f"docker://{space_image}", tarball, registry_creds):
            saved.append(tarball)

    # 2. Base runtime image
    base_image = RUNTIME_IMAGES.get(sdk)
    if base_image:
        tarball = images_dir / f"base-{_safe_filename(base_image)}.tar"
        if skopeo_copy(f"docker://{base_image}", tarball):
            saved.append(tarball)

    # 3. S3 sync init container
    tarball = images_dir / "minio-mc.tar"
    if skopeo_copy(f"docker://{S3_SYNC_IMAGE}", tarball):
        saved.append(tarball)

    # 4. Extra images
    for img in extra_images or []:
        tarball = images_dir / f"{_safe_filename(img)}.tar"
        if skopeo_copy(f"docker://{img}", tarball):
            saved.append(tarball)

    # Write a load script for convenience
    _write_load_script(images_dir, saved)

    return saved


def _write_load_script(images_dir: Path, saved: list[Path]) -> None:
    """Write a helper script to load all saved images into a local registry."""
    if not saved:
        return

    lines = [
        "#!/usr/bin/env bash",
        "# Load all OCI images into a local container registry.",
        "# Usage: ./load-images.sh <registry-url>",
        "# Example: ./load-images.sh localhost:5000",
        "",
        'REGISTRY="${1:?Usage: ./load-images.sh <registry-url>}"',
        "",
        "set -euo pipefail",
        "",
    ]

    for tarball in saved:
        lines.append(f'echo "Loading {tarball.name}..."')
        lines.append(
            f'skopeo copy oci-archive:{tarball.name} '
            f'docker://$REGISTRY/{tarball.stem}:latest'
        )
        lines.append("")

    lines.append('echo "All images loaded into $REGISTRY"')

    script_path = images_dir / "load-images.sh"
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    console.print(f"[green]Load script written:[/] {script_path}")


def _safe_filename(image_ref: str) -> str:
    """Convert an image reference to a safe filename."""
    return image_ref.replace("/", "_").replace(":", "_").replace(".", "-")


def _ref_to_tag(src: str) -> str:
    """Extract a usable tag from a docker:// reference."""
    return src.removeprefix("docker://")
