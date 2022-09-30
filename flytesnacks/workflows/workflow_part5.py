import logging
import os
from typing import Tuple

import flytekit
import mlflow
from flytekit import Secret, task, workflow
from mlflow.tracking import MlflowClient

from flytekit import Resources
from flytesnacks.github import create_pr
from flytesnacks.kubernetes.custom_resource import (
    deploy_custom_resource_imperatively,
    delete_custom_resource_imperatively,
)
from flytesnacks.storage.minio import load_df_from_bucket, save_df_in_bucket


logging.getLogger().setLevel(logging.INFO)

mlflow.set_tracking_uri("http://mlflow-server-service.mlflow.svc.cluster.local:5000")


SECRET_GROUP = "github-access-token"
SECRET_KEY = "token"


@task(cache=True, cache_version="1.3", requests=Resources(cpu="1", mem="500Mi"), limits=Resources(cpu="1", mem="500Mi"))
def prepare_data(test_size: float = 0.3) -> str:
    """Download a dataset, validate it, and return blob storage uri."""

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    """
    Validate data:
    - schema skews
      - E.g. any missing or unexpected columns?
    - values skews
      - Any unexpected `None` values?
      - Any unexpected feature distributions?

    Stop the pipeline if something odd is detected.
    """

    bucket = f"my-s3-bucket/breast_cancer_test_size_{test_size}"
    for df, fn in zip([X_train, X_test, y_train, y_test], ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]):
        uri = f"{bucket}/{fn}"
        save_df_in_bucket(df, uri)

    return bucket


@task(cache=True, cache_version="1.3", requests=Resources(cpu="1", mem="500Mi"), limits=Resources(cpu="1", mem="500Mi"))
def train_model(data_uri: str, n_estimators: int = 100, min_auc: float = 0.95) -> Tuple[str, str, str, str]:
    """
    Train and evaluate model on prepared data at `data_uri`.

    Args:
        data_uri (str): uri of prepared data in blob storage.
        n_estimators (int): number of trees in the random forest
        min_auc (float): minimum required area under the roc curve

    Returns:
        model_name (str): name of the model
        model_version (str): version of the model
    """
    from sklearn.ensemble import RandomForestClassifier

    X_train = load_df_from_bucket(f"{data_uri}/X_train.csv")
    X_test = load_df_from_bucket(f"{data_uri}/X_test.csv")
    y_train = load_df_from_bucket(f"{data_uri}/y_train.csv")
    y_test = load_df_from_bucket(f"{data_uri}/y_test.csv")

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        mlflow.sklearn.eval_and_log_metrics(clf, X_test, y_test, prefix="val_")
        mlflow.log_param("n_estimators", n_estimators)

        """
        Validate model and stop pipeline if performance is lower than expected.
        """
        mlflow_client = MlflowClient()
        metric = mlflow_client.get_metric_history(run.info.run_id, "val_roc_auc_score")
        if metric[0].value < min_auc:
            raise Exception("Stop training due to insufficient performance.")

        model_dir = "model"
        mlflow.sklearn.log_model(
            clf,
            model_dir,
        )
        model_details = mlflow.register_model(
            model_uri=mlflow.get_artifact_uri(model_dir), name="sklearn-random-forest"
        )

    logging.info("Model trained and evaluated")

    return model_details.name, model_details.version, run.info.experiment_id, run.info.run_id


@task(
    cache=True,
    cache_version="1.3",
    container_image="{{.images.explainer.fqn}}:{{.images.default.version}}",
    requests=Resources(mem="1Gi", cpu="0.25"),
    limits=Resources(mem="1Gi", cpu="0.25"),
)
def prepare_explainer(model_name: str, model_version: str) -> str:
    """Prepare an explainer for the model with TreeShap and return the uploaded explainer's uri.
    Args:
        model_name (str): the model to fetch and prepare the explainer for
        model_version (str): the version of the fetched model
    Returns:
        str: uri of the explainer
    """
    # python3.8 pickled files can be opened from python3.7 with pickle5
    import pickle5 as pickle
    from alibi.explainers import TreeShap
    from flytesnacks.storage.minio import download_file, upload_file

    mlflow_client = MlflowClient()

    model_details = mlflow_client.get_model_version(
        name=model_name,
        version=model_version,
    )

    model_base_folder = model_details.source.replace("s3://", "")

    # Need to initialize the model for the explainer
    # using python/mlflow results in numba/cloudpickle errors
    download_file(os.path.join(model_base_folder, "model.pkl"), "./model/model.pkl")

    with open("./model/model.pkl", "rb") as model:
        clf = pickle.load(model)

    # Treeshap doesn't require any training
    # It is able to provide info about what contributed to the classification
    # by comparing the result to a baseline probability
    explainer = TreeShap(clf, task="classification")
    explainer.fit()

    # This artifact must be uploaded and injected into seldon manifest
    local_explainer_path = "model/explainer"
    explainer.save(local_explainer_path)

    # explainerUri should be the folder 'explainer'
    upload_file(
        os.path.join(model_base_folder, "explainer", "explainer.dill"),
        os.path.join(local_explainer_path, "explainer.dill"),
    )
    upload_file(
        os.path.join(model_base_folder, "explainer", "meta.dill"), os.path.join(local_explainer_path, "meta.dill")
    )

    # return the uri
    return os.path.join("s3://", model_base_folder, "explainer")


@task(
    cache=True,
    cache_version="1.3",
    secret_requests=[Secret(key=SECRET_KEY, group=SECRET_GROUP)],
    requests=Resources(cpu="1", mem="500Mi"),
    limits=Resources(cpu="1", mem="500Mi"),
)
def deploy_model(
    model_name: str,
    model_version: str,
    data_uri: str,
    experiment_id: str,
    run_id: str,
    repository: str,
    explainer_uri: str,
) -> str:
    """# noqa D411
    Deploy the model with name `model_name` and version `model_version`.

    Args:
        model_name (str): name of the model in the mlflow model registry
        model_version (str): version of the model in the mlflow model registry
        data_uri (str): uri of prepared data in blob storage.
        experiment_id (str): id of the mlflow experiment the model was trained with
        run_id (str): id of the mlflow run the model was trained with
        repository (str): github repository name to commit seldon deployment manifests
            to in order to deploy them declaratively via GitOps principles
        explainer_uri (str): uri of the explainer artifact
    Returns:
        endpoint_uri (str): the path of the deployed model endpoint.
    """
    from ruamel import yaml

    from flytesnacks.kubernetes.deployment import deploy_deployment_imperatively, delete_deployment_imperatively
    from flytesnacks.kubernetes.service import deploy_service_imperatively, delete_service_imperatively
    from flytesnacks.seldon.manifests import prepare_metrics_server_manifests, prepare_model_manifest
    from flytesnacks.seldon.model_validation import test_asynchronous_feedback, validate_deployed_model, test_explainer

    mlflow_client = MlflowClient()

    model_details = mlflow_client.get_model_version(
        name=model_name,
        version=model_version,
    )

    seldon_deployment_name = model_name + "-" + model_version
    namespace = "inference-test"
    workflow_version = os.environ["FLYTE_INTERNAL_VERSION"]
    artifact_uri = os.path.join(model_details.source.replace("s3://", ""), "model.pkl")

    manifest = prepare_model_manifest(
        namespace=namespace,
        model_name=seldon_deployment_name,
        image_tag=f"k3d-registry.localhost:5000/classifier:{workflow_version}",
        artifact_uri=artifact_uri,
        logger_uri="http://broker-ingress.knative-eventing.svc.cluster.local/seldon-logs/default",
        explainer_type="TreeShap",
        explainer_artifact_uri=explainer_uri,
    )

    manifest_deployment, manifest_service, manifest_trigger = prepare_metrics_server_manifests(
        seldon_deployment_name, namespace, "seldon-logs"
    )

    status = deploy_custom_resource_imperatively(manifest, goal_state=("state", "Available"))
    if status is False:
        raise Exception("Could not deploy model")

    status = deploy_deployment_imperatively(manifest_deployment, timeout=1200)
    status = status and deploy_service_imperatively(manifest_service)
    status = status and deploy_custom_resource_imperatively(
        manifest_trigger, goal_condition={"status": "True", "type": "Ready"}
    )
    if status is False:
        raise Exception("Could not deploy metrics server infrastructure.")

    X_test = load_df_from_bucket(f"{data_uri}/X_test.csv")
    y_test = load_df_from_bucket(f"{data_uri}/y_test.csv")

    model_validated = validate_deployed_model(
        endpoint=f"http://istio-ingressgateway.istio-system.svc.cluster.local:80/seldon/{namespace}/{seldon_deployment_name}/api/v1.0/predictions",  # noqa E501
        model_uri=artifact_uri,
        test_df=X_test,
        request_timeout=1,
        n_batches=5,
        batch_size=5,
        elastic_search_index=f"inference-log-seldon-{namespace}-{seldon_deployment_name}-default",
    )

    feedback_tested = test_asynchronous_feedback(
        host="http://istio-ingressgateway.istio-system.svc.cluster.local:80",
        namespace=namespace,
        seldon_deployment_name=seldon_deployment_name,
        test_inputs=X_test,
        test_labels=y_test,
        request_timeout=1,
        batch_size=5,
    )
    explainer_tested = test_explainer(
        endpoint=f"http://istio-ingressgateway.istio-system.svc.cluster.local:80/seldon/{namespace}/{seldon_deployment_name}-explainer/default/api/v1.0/explain",  # noqa E501
        test_df=X_test,
    )

    status = delete_custom_resource_imperatively(manifest)
    if status is False:
        raise Exception("Could not delete model")

    status = delete_deployment_imperatively(manifest_deployment)
    status = status and delete_service_imperatively(manifest_service)
    status = status and delete_custom_resource_imperatively(manifest_trigger)
    if status is False:
        raise Exception("Could not delete metrics server infrastructure.")

    if not model_validated:
        raise Exception("Model validation failed.")

    if not feedback_tested:
        raise Exception("Feedback mechanism test not passed.")

    if not explainer_tested:
        raise Exception("Explainer test did not pass.")

    run_name = (
        f"{os.environ.get('FLYTE_INTERNAL_EXECUTION_PROJECT')}-"
        f"{os.environ.get('FLYTE_INTERNAL_EXECUTION_DOMAIN')}-"
        f"version-{os.environ.get('FLYTE_INTERNAL_VERSION')}-"
        f"execution-{os.environ.get('FLYTE_INTERNAL_EXECUTION_ID')}"
    )

    production_namespace = "inference"
    manifest["metadata"]["namespace"] = production_namespace
    manifest_deployment, manifest_service, manifest_trigger = prepare_metrics_server_manifests(
        seldon_deployment_name, production_namespace, "seldon-logs"
    )

    endpoint = f"seldon/{production_namespace}/{seldon_deployment_name}/api/v1.0/predictions"

    logging.info(f"Create pull request with model deployment manifests in repository '{repository}'.")
    create_pr(
        repo=repository,
        branch_name=run_name,
        pr_title=f"Deployment run {run_name}",
        pr_description=(f"[Run](http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id})\n" f"`{endpoint}`"),
        files={
            f"infrastructure/inference/{seldon_deployment_name}/model.yaml": yaml.dump(
                manifest, default_flow_style=False
            ),
            f"infrastructure/inference/{seldon_deployment_name}/metrics_server_deployment.yaml": yaml.dump(
                manifest_deployment, default_flow_style=False
            ),
            f"infrastructure/inference/{seldon_deployment_name}/metrics_server_service.yaml": yaml.dump(
                manifest_service, default_flow_style=False
            ),
            f"infrastructure/inference/{seldon_deployment_name}/metrics_server_trigger.yaml": yaml.dump(
                manifest_trigger, default_flow_style=False
            ),
        },
        token=flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_KEY),
    )

    mlflow_client.transition_model_version_stage(
        name=model_details.name, version=model_details.version, stage="Staging"
    )

    return endpoint


@workflow
def pipeline(repository: str, test_size: float = 0.3, n_estimators: int = 100, min_auc: float = 0.95) -> str:
    """Preprocess data, validate data, train model, validate model, and deploy model."""
    data_uri = prepare_data(test_size=test_size)
    model_name, model_version, experiment_id, run_id = train_model(
        data_uri=data_uri, n_estimators=n_estimators, min_auc=min_auc
    )
    explainer_uri = prepare_explainer(
        model_name=model_name,
        model_version=model_version,
    )
    endpoint_uri = deploy_model(
        model_name=model_name,
        model_version=model_version,
        data_uri=data_uri,
        experiment_id=experiment_id,
        run_id=run_id,
        repository=repository,
        explainer_uri=explainer_uri,
    )

    return endpoint_uri


if __name__ == "__main__":
    logging.info(pipeline())
