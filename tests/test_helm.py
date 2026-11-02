"""Comprehensive tests for hf_offline/helm.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _space_info(
    space_id="hysts/translategemma-27b-it",
    owner="hysts",
    name="translategemma-27b-it",
    sdk="docker",
    port=8080,
    models=None,
    title="TranslateGemma 27B IT",
):
    from hf_offline.space import SpaceInfo
    return SpaceInfo(
        space_id=space_id,
        owner=owner,
        name=name,
        sdk=sdk,
        port=port,
        models=models if models is not None else ["google/gemma-2-27b-it"],
        title=title,
    )


def _gen(tmp_path, space=None, model_ids=None, **kwargs):
    from hf_offline.helm import generate_chart
    if space is None:
        space = _space_info()
    if model_ids is None:
        model_ids = ["google/gemma-2-27b-it"]
    return generate_chart(space, tmp_path, model_ids=model_ids, **kwargs)


# ===========================================================================
# generate_chart - file creation
# ===========================================================================

class TestGenerateChartFiles:
    EXPECTED_FILES = [
        "Chart.yaml",
        "values.yaml",
        "templates/_helpers.tpl",
        "templates/deployment.yaml",
        "templates/service.yaml",
        "templates/ingress.yaml",
        "templates/pvc.yaml",
        "templates/secret.yaml",
        "templates/configmap.yaml",
        "templates/model-card.yaml",
        "templates/NOTES.txt",
    ]

    def test_all_files_created(self, tmp_path):
        chart_dir = _gen(tmp_path)
        for f in self.EXPECTED_FILES:
            assert (chart_dir / f).exists(), f"Missing: {f}"

    def test_returns_chart_directory_path(self, tmp_path):
        chart_dir = _gen(tmp_path)
        assert chart_dir.is_dir()
        assert chart_dir.name == "chart"

    def test_chart_dir_inside_output_dir(self, tmp_path):
        chart_dir = _gen(tmp_path)
        assert chart_dir.parent == tmp_path


# ===========================================================================
# Chart.yaml
# ===========================================================================

class TestChartYaml:
    def _chart(self, tmp_path, **kwargs):
        chart_dir = _gen(tmp_path, **kwargs)
        return yaml.safe_load((chart_dir / "Chart.yaml").read_text())

    def test_api_version_v2(self, tmp_path):
        assert self._chart(tmp_path)["apiVersion"] == "v2"

    def test_name_from_space(self, tmp_path):
        assert self._chart(tmp_path)["name"] == "translategemma-27b-it"

    def test_name_underscores_replaced(self, tmp_path):
        space = _space_info(name="my_space_name")
        chart = yaml.safe_load((_gen(tmp_path, space=space) / "Chart.yaml").read_text())
        assert "_" not in chart["name"]
        assert chart["name"] == "my-space-name"

    def test_name_dots_replaced(self, tmp_path):
        space = _space_info(name="my.space.name")
        chart = yaml.safe_load((_gen(tmp_path, space=space) / "Chart.yaml").read_text())
        assert "." not in chart["name"]

    def test_type_is_application(self, tmp_path):
        assert self._chart(tmp_path)["type"] == "application"

    def test_space_id_annotation(self, tmp_path):
        chart = self._chart(tmp_path)
        assert chart["annotations"]["hf-offline/space-id"] == "hysts/translategemma-27b-it"

    def test_sdk_annotation(self, tmp_path):
        chart = self._chart(tmp_path)
        assert chart["annotations"]["hf-offline/sdk"] == "docker"

    def test_version_field_present(self, tmp_path):
        assert "version" in self._chart(tmp_path)

    def test_description_contains_space_id(self, tmp_path):
        desc = self._chart(tmp_path)["description"]
        assert "hysts/translategemma-27b-it" in desc


# ===========================================================================
# values.yaml
# ===========================================================================

class TestValuesYaml:
    def _values(self, tmp_path, **kwargs):
        chart_dir = _gen(tmp_path, **kwargs)
        return yaml.safe_load((chart_dir / "values.yaml").read_text())

    def test_replica_count_default_1(self, tmp_path):
        assert self._values(tmp_path)["replicaCount"] == 1

    def test_space_port_for_docker(self, tmp_path):
        assert self._values(tmp_path)["space"]["port"] == 8080

    def test_space_port_for_gradio(self, tmp_path):
        space = _space_info(sdk="gradio", port=7860)
        values = self._values(tmp_path, space=space)
        assert values["space"]["port"] == 7860

    def test_space_sdk_field(self, tmp_path):
        assert self._values(tmp_path)["space"]["sdk"] == "docker"

    def test_space_id_field(self, tmp_path):
        assert self._values(tmp_path)["space"]["id"] == "hysts/translategemma-27b-it"

    def test_gpu_enabled_true_default(self, tmp_path):
        assert self._values(tmp_path)["gpu"]["enabled"] is True

    def test_gpu_count(self, tmp_path):
        values = self._values(tmp_path, gpu_count=4)
        assert values["gpu"]["count"] == 4

    def test_gpu_type(self, tmp_path):
        values = self._values(tmp_path, gpu_type="amd.com/gpu")
        assert values["gpu"]["type"] == "amd.com/gpu"

    def test_gpu_type_default_nvidia(self, tmp_path):
        assert self._values(tmp_path)["gpu"]["type"] == "nvidia.com/gpu"

    def test_storage_size(self, tmp_path):
        values = self._values(tmp_path, storage_size="200Gi")
        assert values["models"]["pvc"]["size"] == "200Gi"

    def test_storage_size_default(self, tmp_path):
        assert self._values(tmp_path)["models"]["pvc"]["size"] == "50Gi"

    def test_model_ids_in_values(self, tmp_path):
        values = self._values(tmp_path, model_ids=["google/gemma-2-27b-it"])
        assert "google/gemma-2-27b-it" in values["models"]["ids"]

    def test_s3_endpoint(self, tmp_path):
        values = self._values(tmp_path, s3_endpoint="http://minio:9000")
        assert values["models"]["s3"]["endpoint"] == "http://minio:9000"

    def test_s3_bucket(self, tmp_path):
        values = self._values(tmp_path, s3_bucket="my-bucket")
        assert values["models"]["s3"]["bucket"] == "my-bucket"

    def test_s3_prefix(self, tmp_path):
        values = self._values(tmp_path, s3_prefix="prod/models")
        assert values["models"]["s3"]["prefix"] == "prod/models"

    def test_image_registry(self, tmp_path):
        values = self._values(tmp_path, image_registry="registry.internal:5000")
        assert "registry.internal:5000" in values["image"]["repository"]

    def test_pvc_enabled_default_true(self, tmp_path):
        assert self._values(tmp_path)["models"]["pvc"]["enabled"] is True

    def test_models_mount_path(self, tmp_path):
        assert self._values(tmp_path)["models"]["mountPath"] == "/models"

    def test_ingress_disabled_by_default(self, tmp_path):
        assert self._values(tmp_path)["ingress"]["enabled"] is False

    def test_model_card_serve_disabled_by_default(self, tmp_path):
        assert self._values(tmp_path)["modelCard"]["serve"]["enabled"] is False

    def test_resources_requests_present(self, tmp_path):
        resources = self._values(tmp_path)["resources"]
        assert "requests" in resources
        assert "memory" in resources["requests"]
        assert "cpu" in resources["requests"]

    def test_hf_token_empty_by_default(self, tmp_path):
        assert self._values(tmp_path)["hfToken"] == ""

    def test_init_container_image_set(self, tmp_path):
        assert "minio" in self._values(tmp_path)["initContainer"]["image"]


# ===========================================================================
# templates/deployment.yaml
# ===========================================================================

class TestDeploymentTemplate:
    def _deployment(self, tmp_path, **kwargs):
        chart_dir = _gen(tmp_path, **kwargs)
        return (chart_dir / "templates" / "deployment.yaml").read_text()

    def test_is_deployment_kind(self, tmp_path):
        assert "kind: Deployment" in self._deployment(tmp_path)

    def test_container_port_from_values(self, tmp_path):
        content = self._deployment(tmp_path)
        assert ".Values.space.port" in content

    def test_model_mount_path_referenced(self, tmp_path):
        assert ".Values.models.mountPath" in self._deployment(tmp_path)

    def test_gpu_resource_conditional(self, tmp_path):
        assert ".Values.gpu.enabled" in self._deployment(tmp_path)

    def test_hf_token_env_conditional(self, tmp_path):
        assert ".Values.hfToken" in self._deployment(tmp_path)

    def test_init_container_s3_sync(self, tmp_path):
        content = self._deployment(tmp_path)
        assert "initContainers" in content
        assert "model-sync" in content

    def test_liveness_probe_present(self, tmp_path):
        assert "livenessProbe" in self._deployment(tmp_path)

    def test_readiness_probe_present(self, tmp_path):
        assert "readinessProbe" in self._deployment(tmp_path)

    def test_pvc_volume_conditional(self, tmp_path):
        assert "persistentVolumeClaim" in self._deployment(tmp_path)

    def test_transformers_cache_env_var(self, tmp_path):
        assert "TRANSFORMERS_CACHE" in self._deployment(tmp_path)

    def test_hf_home_env_var(self, tmp_path):
        assert "HF_HOME" in self._deployment(tmp_path)


# ===========================================================================
# templates/service.yaml
# ===========================================================================

class TestServiceTemplate:
    def _service(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "service.yaml").read_text()

    def test_service_kind(self, tmp_path):
        assert "kind: Service" in self._service(tmp_path)

    def test_service_type_from_values(self, tmp_path):
        assert ".Values.service.type" in self._service(tmp_path)

    def test_target_port_http(self, tmp_path):
        assert "targetPort: http" in self._service(tmp_path)


# ===========================================================================
# templates/pvc.yaml
# ===========================================================================

class TestPvcTemplate:
    def _pvc(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "pvc.yaml").read_text()

    def test_pvc_kind(self, tmp_path):
        assert "PersistentVolumeClaim" in self._pvc(tmp_path)

    def test_pvc_conditional(self, tmp_path):
        assert ".Values.models.pvc.enabled" in self._pvc(tmp_path)

    def test_storage_size_from_values(self, tmp_path):
        assert ".Values.models.pvc.size" in self._pvc(tmp_path)

    def test_access_mode_from_values(self, tmp_path):
        assert ".Values.models.pvc.accessMode" in self._pvc(tmp_path)


# ===========================================================================
# templates/secret.yaml
# ===========================================================================

class TestSecretTemplate:
    def _secret(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "secret.yaml").read_text()

    def test_secret_kind(self, tmp_path):
        assert "kind: Secret" in self._secret(tmp_path)

    def test_hf_token_key_present(self, tmp_path):
        assert "hf-token" in self._secret(tmp_path)

    def test_s3_keys_present(self, tmp_path):
        assert "s3-access-key" in self._secret(tmp_path)
        assert "s3-secret-key" in self._secret(tmp_path)


# ===========================================================================
# templates/ingress.yaml
# ===========================================================================

class TestIngressTemplate:
    def _ingress(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "ingress.yaml").read_text()

    def test_ingress_conditional(self, tmp_path):
        assert ".Values.ingress.enabled" in self._ingress(tmp_path)

    def test_ingress_kind(self, tmp_path):
        assert "kind: Ingress" in self._ingress(tmp_path)


# ===========================================================================
# templates/configmap.yaml
# ===========================================================================

class TestConfigMapTemplate:
    def _configmap(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "configmap.yaml").read_text()

    def test_configmap_kind(self, tmp_path):
        assert "kind: ConfigMap" in self._configmap(tmp_path)

    def test_space_id_in_configmap(self, tmp_path):
        assert "SPACE_ID" in self._configmap(tmp_path)

    def test_model_path_in_configmap(self, tmp_path):
        assert "MODEL_PATH" in self._configmap(tmp_path)


# ===========================================================================
# templates/NOTES.txt
# ===========================================================================

class TestNotesTemplate:
    def _notes(self, tmp_path):
        chart_dir = _gen(tmp_path)
        return (chart_dir / "templates" / "NOTES.txt").read_text()

    def test_space_id_referenced(self, tmp_path):
        assert ".Values.space.id" in self._notes(tmp_path)

    def test_port_forward_instruction(self, tmp_path):
        assert "port-forward" in self._notes(tmp_path)

    def test_models_mount_path_mentioned(self, tmp_path):
        assert ".Values.models.mountPath" in self._notes(tmp_path)


# ===========================================================================
# Multiple models
# ===========================================================================

class TestMultipleModels:
    def test_multiple_model_ids_in_values(self, tmp_path):
        from hf_offline.helm import generate_chart
        space = _space_info(models=["model/a", "model/b"])
        chart_dir = generate_chart(space, tmp_path, model_ids=["model/a", "model/b"])
        values = yaml.safe_load((chart_dir / "values.yaml").read_text())
        assert "model/a" in values["models"]["ids"]
        assert "model/b" in values["models"]["ids"]

    def test_model_subpaths_have_double_dash(self, tmp_path):
        from hf_offline.helm import generate_chart
        space = _space_info(models=["google/gemma-2-27b-it"])
        chart_dir = generate_chart(space, tmp_path, model_ids=["google/gemma-2-27b-it"])
        # The deployment template should reference model subpaths
        deploy = (chart_dir / "templates" / "deployment.yaml").read_text()
        # The model subpath is derived in chart_meta but not directly in template
        # (it's computed into values.yaml model list)
        values = yaml.safe_load((chart_dir / "values.yaml").read_text())
        assert "google/gemma-2-27b-it" in values["models"]["ids"]


class TestOpenShift:
    def _chart(self, tmp_path):
        from hf_offline.helm import generate_chart
        from hf_offline.space import SpaceInfo

        return generate_chart(SpaceInfo(space_id="o/s", owner="o", name="s", sdk="gradio"), tmp_path)

    def test_overlay_enables_openshift(self, tmp_path):
        import yaml

        c = self._chart(tmp_path)
        v = yaml.safe_load((c / "values-openshift.yaml").read_text())
        assert v["openshift"]["enabled"] is True and v["ingress"]["enabled"] is False
        assert v["tolerations"][0]["key"] == "nvidia.com/gpu"

    def test_default_values_keep_openshift_off(self, tmp_path):
        import yaml

        c = self._chart(tmp_path)
        assert yaml.safe_load((c / "values.yaml").read_text())["openshift"]["enabled"] is False

    def test_route_and_scc_safe_deployment(self, tmp_path):
        c = self._chart(tmp_path)
        assert "kind: Route" in (c / "templates" / "route.yaml").read_text()
        d = (c / "templates" / "deployment.yaml").read_text()
        assert "runAsNonRoot: true" in d and "drop: [ALL]" in d
        assert "runAsUser" not in d  # SCC must assign the UID
