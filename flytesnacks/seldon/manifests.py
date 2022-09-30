"""Create seldon deployment manifests."""
from typing import Optional, Tuple


seldon_deployment_template = """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  namespace: {{namespace}}
  name: {{model_name}}
spec:
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: {{image_tag}}
          imagePullPolicy: Always
    graph:
      name: classifier
      type: MODEL
      {% if logger_uri is defined %}
      logger:
        url: {{logger_uri}}
        mode: all
      {% endif %}
      children: []
      parameters:
      - name: model_uri
        type: STRING
        value: "{{artifact_uri}}"
    {% if explainer is defined %}
    explainer:
      type: {{explainer.type}}
      envSecretRefName: seldon-init-container-secret
      modelUri: {{explainer.artifact_uri}}
      containerSpec:
        image: seldonio/alibiexplainer:1.12.0
        name: explainer
    {% endif %}
    name: default
    replicas: 1
"""


metrics_server_trigger_template = """apiVersion: eventing.knative.dev/v1
kind: Trigger
metadata:
  name: {{model_name}}-metrics-trigger
  namespace: {{namespace_trigger}}
spec:
  broker: default
  filter:
    attributes:
      type: io.seldon.serving.feedback
  subscriber:
    uri: http://seldon-{{model_name}}-metrics.{{namespace}}.svc.cluster.local:80
"""


metrics_server_service_template = """apiVersion: v1
kind: Service
metadata:
  namespace: {{namespace}}
  name: seldon-{{model_name}}-metrics
  labels:
    app: seldon-{{model_name}}-metrics
spec:
  selector:
    app: seldon-{{model_name}}-metrics
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
"""


metrics_server_deployment_template = """apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: {{namespace}}
  name: seldon-{{model_name}}-metrics
  labels:
    app: seldon-{{model_name}}-metrics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: seldon-{{model_name}}-metrics
  template:
    metadata:
      annotations:
        prometheus.io/path: /v1/metrics
        prometheus.io/scrape: "true"
      labels:
        app: seldon-{{model_name}}-metrics
    spec:
      securityContext:
        runAsUser: 8888
      containers:
      - name: user-container
        image: seldonio/alibi-detect-server:1.10.0
        imagePullPolicy: IfNotPresent
        args:
        - --model_name
        - multiclassserver
        - --http_port
        - '8080'
        - --protocol
        - seldonfeedback.http
        - --storage_uri
        - "adserver.cm_models.multiclass_one_hot.MulticlassOneHot"
        - --event_type
        - io.seldon.serving.feedback.metrics
        - --event_source
        - io.seldon.serving.feedback
        - --elasticsearch_uri
        - http://elasticsearch-master.seldon-logs.svc.cluster.local:9200
        - MetricsServer
        env:
        - name: "SELDON_DEPLOYMENT_ID"
          value: "{{model_name}}"
        - name: "PREDICTIVE_UNIT_ID"
          value: "classifier"
        - name: "PREDICTIVE_UNIT_IMAGE"
          value: "alibi-detect-server:1.10.0"
        - name: "PREDICTOR_ID"
          value: "default"
        ports:
        - containerPort: 8080
          name: metrics
          protocol: TCP"""


def prepare_model_manifest(
    namespace: str,
    model_name: str,
    image_tag: str,
    artifact_uri: str,
    logger_uri: Optional[str] = None,
    explainer_type: Optional[str] = None,
    explainer_artifact_uri: Optional[str] = None,
) -> dict:
    """
    Prepare a seldon deployment manifest by rendering a jinja2 template.

    Args:
        namespace (str): kubernetes namespace the model is deployed to
        model_name (str): name of the seldon deployment
        image_tag (str): tag of the docker image to be used in the seldon deployment
        artifact_uri (str): mlflow tracking server uri pointing to a logged model

    Kwargs:
        logger_uri (str): request logger endpoint uri
        explainer_type (str): type of the seldon explainer:
            https://docs.seldon.io/projects/seldon-core/en/latest/analytics/explainers.html
            Required when providing an explainer artifact uri.
        explainer_artifact_uri (str): uri of the explainer model. Required when
            providing an explainer type.

    Returns:
        Seldon deployment manifest as a dict
    """
    if explainer_type is not None:
        assert explainer_artifact_uri is not None, "Provide an explainer artifact uri when providing an explainer type."
    if explainer_artifact_uri is not None:
        assert explainer_type is not None, "Provide an explainer type when providing an explainer artifact uri."

    from jinja2 import Template
    from ruamel import yaml

    data = {"namespace": namespace, "model_name": model_name, "image_tag": image_tag, "artifact_uri": artifact_uri}
    if logger_uri is not None:
        data["logger_uri"] = logger_uri
    if explainer_type is not None:
        data["explainer"] = {
            "type": explainer_type,
            "artifact_uri": explainer_artifact_uri,
        }

    seldon_deplyoment_template = Template(seldon_deployment_template)

    return yaml.safe_load(seldon_deplyoment_template.render(data))


def prepare_metrics_server_manifests(
    model_name: str,
    namespace_model: str,
    namespace_trigger: str,
) -> Tuple[dict, dict, dict]:
    """
    Prepare a deployment, service, and knative-eventing trigger manifest for a seldon metrics server.

    Args:
        model_name (str): name of the seldon deployment
        namespace_model (str): kubernetes namespace the model is deployed to
        namespace_trigger (str): kubernetes namespace the knative
            eventing trigger should be deployed to

    Returns:
        A tuple of three dictionaries: deployment, service, and trigger manifests
    """
    from jinja2 import Template
    from ruamel import yaml

    data = {
        "namespace": namespace_model,
        "model_name": model_name,
        "namespace_trigger": namespace_trigger,
    }

    deplyoment_template = Template(metrics_server_deployment_template)
    service_template = Template(metrics_server_service_template)
    trigger_template = Template(metrics_server_trigger_template)

    return (
        yaml.safe_load(deplyoment_template.render(data)),
        yaml.safe_load(service_template.render(data)),
        yaml.safe_load(trigger_template.render(data)),
    )
