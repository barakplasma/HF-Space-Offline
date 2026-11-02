# HF-Space-Offline

Download HuggingFace Spaces for offline deployment on air-gapped Kubernetes GPU clusters.

Produces a single self-contained bundle folder (inspired by [Pinokio](https://pinokio.computer)) containing everything needed to deploy a HuggingFace Space without internet access:

```
bundle-myorg--my-space/
├── app/              # Space application code + Dockerfile
├── models/           # Model weights (separate dir for PVC/S3 mount)
│   ├── org--model/   # Each model in its own subfolder
│   └── manifest.yaml
├── images/           # OCI container image tarballs (via skopeo)
│   ├── base-python_3-10-slim.tar
│   ├── minio-mc.tar
│   └── load-images.sh
├── chart/            # Complete Helm chart for K8s deployment
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── model-card.html   # Rendered model card with metadata
└── manifest.json     # Bundle metadata and deploy instructions
```

## Install

```bash
pip install -e .
```

### System dependencies

- **skopeo** - Required for exporting container images ([install](https://github.com/containers/skopeo/blob/main/install.md))
- **helm** - Required for deploying the chart
- Alternatives: [Zarf](https://docs.zarf.dev/) or [Hauler](https://hauler.dev/) for full air-gap packaging

## Usage

### Basic download

```bash
hf-offline download gradio/chatbot
```

### With a gated model (requires HF token)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx

hf-offline download TheBloke/Llama-2-7b-Chat-GPTQ \
  --model meta-llama/Llama-2-7b-chat-hf
```

### Full production setup

```bash
hf-offline download myorg/my-space \
  --model myorg/my-model \
  --hf-token hf_xxxxx \
  --s3-endpoint http://minio.storage:9000 \
  --s3-bucket models \
  --gpu-count 2 \
  --gpu-type nvidia.com/gpu \
  --storage-size 100Gi \
  --image-registry registry.internal:5000 \
  --output ./my-bundle
```

### Inspect an existing bundle

```bash
hf-offline inspect ./my-bundle
```

## Deployment workflow

### 1. Transfer the bundle

Copy the entire bundle folder to your air-gapped environment (USB drive, S3 sync, etc.).

### 2. Load container images

```bash
cd bundle-myorg--my-space/images
./load-images.sh registry.internal:5000
```

### 3. Build and push the space image

```bash
cd bundle-myorg--my-space/app
docker build -t registry.internal:5000/my-space:latest .
docker push registry.internal:5000/my-space:latest
```

### 4. Upload models to S3

```bash
mc alias set s3store http://minio:9000 ACCESS_KEY SECRET_KEY
mc cp --recursive bundle-myorg--my-space/models/ s3store/models/
```

### 5. Configure and deploy

Edit `chart/values.yaml` with your environment-specific settings:

```yaml
image:
  repository: registry.internal:5000/my-space
  tag: latest

models:
  s3:
    endpoint: http://minio:9000
    bucket: models
    accessKey: your-access-key
    secretKey: your-secret-key

gpu:
  enabled: true
  type: nvidia.com/gpu
  count: 1
```

Then deploy:

```bash
helm install my-space bundle-myorg--my-space/chart/
```

## OpenShift / OpenShift AI (disconnected)

The chart ships a `values-openshift.yaml` overlay. It swaps the Ingress for a `Route`, adds
restricted-SCC-safe security contexts (non-root, arbitrary UID, all capabilities dropped,
no `runAsUser`/`fsGroup`), and tolerates the usual `nvidia.com/gpu` taint. Generated
Dockerfiles are arbitrary-UID safe (group-0 writable, `HOME=/tmp/home`, `USER 1001`).

```bash
# 1. Mirror images into the cluster registry (app image + minio/mc, plus nginx-unprivileged if serving the model card)
# 2. Upload models/ to your S3 (an OpenShift AI data connection works: pass its Secret name)
helm install myapp chart/ -f chart/values-openshift.yaml \
  --set image.repository=image-registry.openshift-image-registry.svc:5000/<ns>/myapp \
  --set initContainer.image=<mirror>/minio/mc:latest \
  --set models.s3.endpoint=http://minio.<ns>.svc:9000 \
  --set models.s3.existingSecret=<data-connection-secret> \
  --set 'imagePullSecrets[0].name=<pull-secret>'
```

This deploys the Space as a regular Deployment + Route inside an OpenShift AI project. It is
not a KServe `InferenceService`: Gradio/Streamlit apps are UIs, not model servers. Spaces that
ship their own Dockerfile are kept as-is and must also run as a non-root, arbitrary UID.

## Helm chart features

- **Init container** syncs models from S3 to PVC on startup (using `minio/mc`)
- **PVC** for persistent model storage
- **GPU scheduling** with configurable resource type and count
- **Secrets** for HF token and S3 credentials
- **Ingress** support (optional)
- **Model card serving** via nginx sidecar (optional, enable `modelCard.serve.enabled`)

## CLI reference

```
hf-offline download SPACE_ID [OPTIONS]

Options:
  -m, --model TEXT           Model ID(s) to download (repeatable)
  -t, --hf-token TEXT        HuggingFace API token (or HF_TOKEN env)
  -o, --output TEXT          Output directory
  --s3-endpoint TEXT         S3-compatible endpoint URL
  --s3-bucket TEXT           S3 bucket name (default: models)
  --s3-prefix TEXT           S3 key prefix
  --gpu-count INT            GPU count (default: 1)
  --gpu-type TEXT            GPU resource type (default: nvidia.com/gpu)
  --storage-size TEXT        PVC size (default: 50Gi)
  --image-registry TEXT      Registry URL for Helm chart (default: localhost:5000)
  --space-image TEXT         Pre-built space image ref to save
  --extra-image TEXT         Additional images to export (repeatable)
  --skip-images              Skip container image export
```

## Gated model support

Some models on HuggingFace require accepting terms and providing an API token. To download gated models:

1. Accept the model's terms on the HuggingFace website
2. Create an API token at https://huggingface.co/settings/tokens
3. Pass it via `--hf-token` or the `HF_TOKEN` environment variable

## Architecture

```
hf_offline/
├── cli.py          # Click CLI (download, inspect commands)
├── bundle.py       # Orchestrator - ties everything together
├── space.py        # Space metadata, cloning, Dockerfile generation
├── models.py       # Model downloading with gated support
├── images.py       # Container image export via skopeo
├── helm.py         # Helm chart generation
└── model_card.py   # Model card HTML rendering
```
