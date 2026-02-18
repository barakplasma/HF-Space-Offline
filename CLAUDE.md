# CLAUDE.md

## Project overview

hf-offline is a Python CLI tool that downloads HuggingFace Spaces and bundles them for offline deployment on air-gapped Kubernetes GPU clusters. It produces a single self-contained folder with app code, model weights, OCI container images, a Helm chart, and a rendered model card.

## Tech stack

- Python 3.10+, built with hatchling
- CLI: click, rich (for terminal output)
- HuggingFace: huggingface-hub (API + downloads)
- YAML: pyyaml
- Markdown: markdown (for model card rendering)
- Container images: skopeo (system dependency, shelled out via subprocess)
- Deployment: Helm 3 charts with Go templates

## Project structure

```
hf_offline/
├── cli.py          # Click CLI entry point (download, inspect commands)
├── bundle.py       # Orchestrator - 7-step pipeline tying modules together
├── space.py        # HF Space metadata, repo cloning, Dockerfile generation
├── models.py       # Model downloading with gated model support
├── images.py       # Container image export via skopeo
├── helm.py         # Helm chart generation (11 template files)
└── model_card.py   # Model card markdown → self-contained HTML
```

## Commands

```bash
# Install in development mode
pip install -e .

# Run the CLI
hf-offline download <space_id> [OPTIONS]
hf-offline inspect <bundle_dir>

# Run tests
pytest

# Lint
ruff check hf_offline/
```

## Key design decisions

- **Pinokio-inspired**: one folder = one app, everything self-contained with a manifest.json
- **Models separate from app**: models/ directory is distinct from app/ — intended for PVC mount backed by generic S3
- **Generic S3**: no vendor lock-in, works with MinIO, SeaweedFS, any S3-compatible endpoint
- **skopeo for images**: saves OCI archives; Zarf/Hauler mentioned as alternatives
- **Helm is the deploy mechanism**: no custom deploy CLI, just `helm install`
- **Gradio + Streamlit focus**: the 80% of HF Spaces; generates Dockerfiles for these SDKs
- **Gated models**: pass HF_TOKEN env var or --hf-token flag

## Helm chart architecture

The generated chart includes:
- Deployment with init container (minio/mc) that syncs models from S3 → PVC
- PVC for model storage, GPU resource scheduling (nvidia.com/gpu default)
- Secrets for HF token and S3 credentials
- Optional Ingress, optional model card serving via nginx sidecar

## Code conventions

- Type hints throughout (from __future__ import annotations)
- Dataclasses for structured data (SpaceInfo, ModelInfo)
- rich.console for user-facing output
- No Jinja2 — Helm templates are Go template strings written directly from Python
- YAML frontmatter parsing for HF README metadata
