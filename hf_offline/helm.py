"""Helm chart generation for offline HuggingFace Space deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from hf_offline.space import SpaceInfo

console = Console()


def generate_chart(
    space_info: SpaceInfo,
    output_dir: Path,
    *,
    model_ids: list[str] | None = None,
    s3_endpoint: str = "",
    s3_bucket: str = "models",
    s3_prefix: str = "",
    gpu_count: int = 1,
    gpu_type: str = "nvidia.com/gpu",
    storage_size: str = "50Gi",
    image_registry: str = "localhost:5000",
    env: dict[str, str] | None = None,
) -> Path:
    """Generate a complete Helm chart for the space deployment.

    The chart includes:
    - Deployment with init container for S3 model sync
    - PVC for model storage
    - Service and optional Ingress
    - Secrets for HF token and S3 credentials
    - ConfigMap for app environment
    - Optional model card serving via nginx
    """
    chart_dir = output_dir / "chart"
    templates_dir = chart_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    safe_name = space_info.name.lower().replace("_", "-").replace(".", "-")
    model_ids = model_ids or space_info.models

    # Compute the model subpaths for the init container to sync
    model_subpaths = ["models--" + mid.replace("/", "--") for mid in model_ids]

    chart_meta = {
        "name": safe_name,
        "space_id": space_info.space_id,
        "sdk": space_info.sdk,
        "port": space_info.port,
        "image_registry": image_registry,
        "model_ids": model_ids,
        "model_subpaths": model_subpaths,
        "env": dict(env or {}),
        "s3_endpoint": s3_endpoint,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "storage_size": storage_size,
    }

    # Write all chart files
    _write_chart_yaml(chart_dir, chart_meta)
    _write_values_yaml(chart_dir, chart_meta)
    _write_helpers_tpl(templates_dir, chart_meta)
    _write_deployment(templates_dir, chart_meta)
    _write_service(templates_dir, chart_meta)
    _write_ingress(templates_dir, chart_meta)
    _write_pvc(templates_dir, chart_meta)
    _write_secrets(templates_dir, chart_meta)
    _write_configmap(templates_dir, chart_meta)
    _write_model_card_serving(templates_dir, chart_meta)
    _write_notes(templates_dir, chart_meta)

    console.print(f"[green]Helm chart generated at:[/] {chart_dir}")
    return chart_dir


# ---------------------------------------------------------------------------
# Chart.yaml
# ---------------------------------------------------------------------------

def _write_chart_yaml(chart_dir: Path, meta: dict) -> None:
    content = {
        "apiVersion": "v2",
        "name": meta["name"],
        "description": f"Offline deployment of HuggingFace Space {meta['space_id']}",
        "type": "application",
        "version": "0.1.0",
        "appVersion": "1.0.0",
        "keywords": ["huggingface", "ml", "gpu", "offline"],
        "annotations": {
            "hf-offline/space-id": meta["space_id"],
            "hf-offline/sdk": meta["sdk"],
        },
    }
    (chart_dir / "Chart.yaml").write_text(
        yaml.dump(content, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# values.yaml
# ---------------------------------------------------------------------------

def _write_values_yaml(chart_dir: Path, meta: dict) -> None:
    values: dict[str, Any] = {
        "replicaCount": 1,

        "image": {
            "repository": f"{meta['image_registry']}/{meta['name']}",
            "tag": "latest",
            "pullPolicy": "IfNotPresent",
        },

        "space": {
            "id": meta["space_id"],
            "sdk": meta["sdk"],
            "port": meta["port"],
        },

        "models": {
            "ids": meta["model_ids"],
            "mountPath": "/models",
            "s3": {
                "endpoint": meta["s3_endpoint"],
                "bucket": meta["s3_bucket"],
                "prefix": meta["s3_prefix"],
                "accessKey": "",
                "secretKey": "",
            },
            "pvc": {
                "enabled": True,
                "size": meta["storage_size"],
                "storageClass": "",
                "accessMode": "ReadWriteOnce",
            },
        },

        "gpu": {
            "enabled": True,
            "type": meta["gpu_type"],
            "count": meta["gpu_count"],
        },

        "resources": {
            "requests": {
                "memory": "8Gi",
                "cpu": "2",
            },
            "limits": {
                "memory": "16Gi",
            },
        },

        "service": {
            "type": "ClusterIP",
            "port": 80,
        },

        "ingress": {
            "enabled": False,
            "className": "",
            "annotations": {},
            "host": "",
            "tls": [],
        },

        "hfToken": "",

        "env": meta.get("env", {}),

        "modelCard": {
            "serve": {
                "enabled": False,
                "port": 8080,
            },
        },

        "initContainer": {
            "image": "minio/mc:latest",
        },

        "nodeSelector": {},
        "tolerations": [],
        "affinity": {},
    }

    (chart_dir / "values.yaml").write_text(
        yaml.dump(values, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# templates/_helpers.tpl
# ---------------------------------------------------------------------------

def _write_helpers_tpl(templates_dir: Path, meta: dict) -> None:
    content = """\
{{/*
Expand the name of the chart.
*/}}
{{- define "space.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "space.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "space.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "space.labels" -}}
helm.sh/chart: {{ include "space.chart" . }}
{{ include "space.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "space.selectorLabels" -}}
app.kubernetes.io/name: {{ include "space.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
"""
    (templates_dir / "_helpers.tpl").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/deployment.yaml
# ---------------------------------------------------------------------------

def _write_deployment(templates_dir: Path, meta: dict) -> None:
    content = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "space.fullname" . }}
  labels:
    {{- include "space.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "space.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "space.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}

      # Init container: sync models from S3 to the PVC
      {{- if and .Values.models.s3.endpoint .Values.models.pvc.enabled }}
      initContainers:
        - name: model-sync
          image: {{ .Values.initContainer.image }}
          command:
            - /bin/sh
            - -c
            - |
              mc alias set s3store {{ .Values.models.s3.endpoint }} $(S3_ACCESS_KEY) $(S3_SECRET_KEY)
              mc mirror --overwrite s3store/{{ .Values.models.s3.bucket }}/{{ .Values.models.s3.prefix }} /models/
          env:
            - name: S3_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "space.fullname" . }}-secrets
                  key: s3-access-key
            - name: S3_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "space.fullname" . }}-secrets
                  key: s3-secret-key
          volumeMounts:
            - name: models
              mountPath: /models
      {{- end }}

      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.space.port }}
              protocol: TCP
          env:
            - name: TRANSFORMERS_CACHE
              value: {{ .Values.models.mountPath }}
            - name: HF_HOME
              value: {{ .Values.models.mountPath }}
            - name: HF_HUB_CACHE
              value: {{ .Values.models.mountPath }}
            - name: HF_HUB_OFFLINE
              value: "1"
            {{- if .Values.hfToken }}
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: {{ include "space.fullname" . }}-secrets
                  key: hf-token
            {{- end }}
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          volumeMounts:
            - name: models
              mountPath: {{ .Values.models.mountPath }}
          resources:
            requests:
              {{- toYaml .Values.resources.requests | nindent 14 }}
            limits:
              {{- if .Values.gpu.enabled }}
              {{ .Values.gpu.type }}: {{ .Values.gpu.count }}
              {{- end }}
              {{- if .Values.resources.limits }}
              {{- toYaml .Values.resources.limits | nindent 14 }}
              {{- end }}
          livenessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 30
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5

      volumes:
        - name: models
          {{- if .Values.models.pvc.enabled }}
          persistentVolumeClaim:
            claimName: {{ include "space.fullname" . }}-models
          {{- else }}
          emptyDir: {}
          {{- end }}
"""
    (templates_dir / "deployment.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/service.yaml
# ---------------------------------------------------------------------------

def _write_service(templates_dir: Path, meta: dict) -> None:
    content = """\
apiVersion: v1
kind: Service
metadata:
  name: {{ include "space.fullname" . }}
  labels:
    {{- include "space.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "space.selectorLabels" . | nindent 4 }}
"""
    (templates_dir / "service.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/ingress.yaml
# ---------------------------------------------------------------------------

def _write_ingress(templates_dir: Path, meta: dict) -> None:
    content = """\
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "space.fullname" . }}
  labels:
    {{- include "space.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    - host: {{ .Values.ingress.host | quote }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ include "space.fullname" . }}
                port:
                  name: http
{{- end }}
"""
    (templates_dir / "ingress.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/pvc.yaml
# ---------------------------------------------------------------------------

def _write_pvc(templates_dir: Path, meta: dict) -> None:
    content = """\
{{- if .Values.models.pvc.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "space.fullname" . }}-models
  labels:
    {{- include "space.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.models.pvc.accessMode }}
  {{- if .Values.models.pvc.storageClass }}
  storageClassName: {{ .Values.models.pvc.storageClass | quote }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.models.pvc.size }}
{{- end }}
"""
    (templates_dir / "pvc.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/secret.yaml
# ---------------------------------------------------------------------------

def _write_secrets(templates_dir: Path, meta: dict) -> None:
    content = """\
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "space.fullname" . }}-secrets
  labels:
    {{- include "space.labels" . | nindent 4 }}
type: Opaque
stringData:
  {{- if .Values.hfToken }}
  hf-token: {{ .Values.hfToken | quote }}
  {{- end }}
  {{- if .Values.models.s3.accessKey }}
  s3-access-key: {{ .Values.models.s3.accessKey | quote }}
  s3-secret-key: {{ .Values.models.s3.secretKey | quote }}
  {{- end }}
"""
    (templates_dir / "secret.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/configmap.yaml
# ---------------------------------------------------------------------------

def _write_configmap(templates_dir: Path, meta: dict) -> None:
    content = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "space.fullname" . }}-config
  labels:
    {{- include "space.labels" . | nindent 4 }}
data:
  SPACE_ID: {{ .Values.space.id | quote }}
  SPACE_SDK: {{ .Values.space.sdk | quote }}
  MODEL_PATH: {{ .Values.models.mountPath | quote }}
"""
    (templates_dir / "configmap.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/model-card.yaml  (optional nginx serving the model card HTML)
# ---------------------------------------------------------------------------

def _write_model_card_serving(templates_dir: Path, meta: dict) -> None:
    content = """\
{{- if .Values.modelCard.serve.enabled }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "space.fullname" . }}-model-card
  labels:
    {{- include "space.labels" . | nindent 4 }}
data:
  index.html: |
    {{- .Files.Get "model-card.html" | nindent 4 }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "space.fullname" . }}-model-card
  labels:
    {{- include "space.labels" . | nindent 4 }}
    app.kubernetes.io/component: model-card
spec:
  replicas: 1
  selector:
    matchLabels:
      {{- include "space.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: model-card
  template:
    metadata:
      labels:
        {{- include "space.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: model-card
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
          volumeMounts:
            - name: html
              mountPath: /usr/share/nginx/html
      volumes:
        - name: html
          configMap:
            name: {{ include "space.fullname" . }}-model-card
---
apiVersion: v1
kind: Service
metadata:
  name: {{ include "space.fullname" . }}-model-card
  labels:
    {{- include "space.labels" . | nindent 4 }}
    app.kubernetes.io/component: model-card
spec:
  type: ClusterIP
  ports:
    - port: {{ .Values.modelCard.serve.port }}
      targetPort: 80
      protocol: TCP
  selector:
    {{- include "space.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: model-card
{{- end }}
"""
    (templates_dir / "model-card.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# templates/NOTES.txt
# ---------------------------------------------------------------------------

def _write_notes(templates_dir: Path, meta: dict) -> None:
    content = """\
=== HF Space Offline Deployment ===

Space: {{ .Values.space.id }}
SDK:   {{ .Values.space.sdk }}

1. Get the application URL:
{{- if .Values.ingress.enabled }}
  http://{{ .Values.ingress.host }}
{{- else }}
  kubectl port-forward svc/{{ include "space.fullname" . }} 8080:{{ .Values.service.port }}
  Then open: http://localhost:8080
{{- end }}

2. Models are mounted at: {{ .Values.models.mountPath }}

{{- if and .Values.models.s3.endpoint .Values.models.pvc.enabled }}
3. Models will be synced from S3 on startup:
   Endpoint: {{ .Values.models.s3.endpoint }}
   Bucket:   {{ .Values.models.s3.bucket }}

   To manually upload models to S3:
     mc alias set s3store {{ .Values.models.s3.endpoint }} <access-key> <secret-key>
     mc cp --recursive models/ s3store/{{ .Values.models.s3.bucket }}/
{{- end }}

{{- if .Values.modelCard.serve.enabled }}
4. Model card is served at:
   kubectl port-forward svc/{{ include "space.fullname" . }}-model-card {{ .Values.modelCard.serve.port }}:{{ .Values.modelCard.serve.port }}
{{- end }}
"""
    (templates_dir / "NOTES.txt").write_text(content, encoding="utf-8")
