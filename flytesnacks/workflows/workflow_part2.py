import logging
from typing import Tuple

import mlflow
from flytekit import task, workflow
from mlflow.tracking import MlflowClient

from flytesnacks.storage.minio import load_df_from_bucket, save_df_in_bucket


logging.getLogger().setLevel(logging.INFO)

mlflow.set_tracking_uri("http://mlflow-server-service.mlflow.svc.cluster.local:5000")


@task(cache=True, cache_version="1.0")
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


@task(cache=True, cache_version="1.0")
def train_model(data_uri: str, n_estimators: int = 100, min_auc: float = 0.95) -> Tuple[str, str]:
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
        client = MlflowClient()
        metric = client.get_metric_history(run.info.run_id, "val_roc_auc_score")
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

    return model_details.name, model_details.version


@task(cache=True, cache_version="1.0")
def deploy_model(model_name: str, model_version: str) -> str:
    """
    Deploy the model with name `model_name` and version `model_version`.

    Args:
        model_name (str): name of the model in the mlflow model registry
        model_version (str): version of the model in the mlflow model registry
    Returns:
        endpoint_uri (str): the path of the deployed model endpoint.
    """
    client = MlflowClient()

    model_details = client.get_model_version(
        name=model_name,
        version=model_version,
    )

    """
    Deploy model and validate that the deployment worked.
    """

    client.transition_model_version_stage(name=model_details.name, version=model_details.version, stage="Staging")

    return "endpoint_uri"


@workflow
def pipeline(test_size: float = 0.3, n_estimators: int = 100, min_auc: float = 0.95) -> str:
    """Preprocess data, validate data, train model, validate model, and deploy model."""
    data_uri = prepare_data(test_size=test_size)
    model_name, model_version = train_model(data_uri=data_uri, n_estimators=n_estimators, min_auc=min_auc)
    endpoint_uri = deploy_model(model_name=model_name, model_version=model_version)

    return endpoint_uri


if __name__ == "__main__":
    logging.info(pipeline())
