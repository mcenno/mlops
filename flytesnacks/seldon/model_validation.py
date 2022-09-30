import logging
import requests
import time
from typing import List

import pandas as pd


logging.getLogger().setLevel(logging.INFO)


def get_docs_from_index(
    index: str,
    host: str = "http://elasticsearch-master.seldon-logs.svc.cluster.local:9200",
    max_docs: int = 1e3,
) -> List:
    """
    Return all documents in an elasticsearch index.

    Args:
        index (str): the elasticsearch index to search
    Kwargs:
        host (str): uri of the elasticsearch endpoint
        max_docs (str): maximum number of documents to return

    Returns:
        A list of elasticsearch documents. An empty list of the index
        is empty or does not exist.
    """
    import elasticsearch
    from elasticsearch import Elasticsearch

    es = Elasticsearch(host)
    try:
        res = es.search(index=index, body={"size": max_docs, "query": {"match_all": {}}})
    except elasticsearch.exceptions.NotFoundError:
        return []

    return res["hits"]["hits"]


def validate_deployed_model(
    endpoint: str,
    model_uri: str,
    test_df: pd.core.frame.DataFrame,
    request_timeout: float = 1.0,
    batch_size: int = 10,
    n_batches: int = 20,
    elastic_search_index: str = None,
    elastic_search_logging_timeout: int = 60,
) -> bool:
    """Validate a deployed seldon model.

    The model is validated regarding:
    - correctness of predictions
    - inference speed
    - request logging into elasticsearch (optional)

    To do so, the model is loaded from the mlflow tracking server.
    Predictions are generated for the test data from `test_df` both
    using the model deployed with seldon as well as with the version
    that is locally loaded from the mlflow tracking server. The
    predictions have to match.

    Additionally, it is asserted that the predictions are faster than
    `request_timeout` in seconds.

    Optionally, if an elasticsearch index is provided, it is asserted
    that all prediction requests are logged correctly in a specified
    amount of time.

    Args:
        endpoint (str): uri of the deployed model. Typically the service name
           of the istio ingressgateway with the path derived from the model's
           namespace and its name.
        model_uri (str): uri of the model in the mlflow tracking server
        test_df (DataFrame): pandas dataframe containing test data
    Kwargs:
        request_timeout (int): predictions cannot take longer than this timeout in
            seconds for the model to pass validation
        batch_size (int): number of samples to run inference on at once
        n_batches (int): number of batches to validate the model on
        elastic_search_index (str): elastic search index that model requests
            are logged into. It is asserted that the test requests can be found
            in this index.
        elastic_search_logging_timeout (int): maximum number of seconds allowed
            for all requests to be logged into elasticsearch.

    Returns:
        True if the model passes validation, False otherwise

    """
    import numpy as np
    import pickle
    import requests
    from minio import Minio

    client = Minio("minio.flyte:9000", access_key="minio", secret_key="miniostorage", secure=False)

    # Load model from tracking server
    bucket, fp = model_uri.split("/", 1)
    obj = client.get_object(
        bucket,
        fp,
    )
    model = pickle.load(obj)

    results = []

    if elastic_search_index is not None:
        n_prior_docs_in_index = len(get_docs_from_index(elastic_search_index))

        logging.info(
            f"The elasticsearch index already contains {n_prior_docs_in_index} documents before sending the predictions"
        )

    """
    Assert that predictions returned by the model served with seldon
    are equivalent to predictions obtained from a local version of the
    model.
    """

    for _ in range(n_batches):
        sample = test_df.sample(batch_size)
        logging.info(f"Comparing local and deployed model on batch of shape {sample.shape}")

        local_predictions = model.predict_proba(sample)

        data = {"data": {"names": test_df.columns.to_list(), "ndarray": sample.values.tolist()}}

        try:
            response = requests.post(endpoint, json=data, timeout=request_timeout)
        except requests.ReadTimeout:
            logging.warning("Model not fast enough")
            return False

        if not response.status_code == requests.codes["ok"]:
            logging.warning("Prediction request failed.")
            return False

        if response.json().get("data") is None or response.json().get("data").get("ndarray") is None:
            logging.warning("Response not as expected")
            return False

        remote_predictions = np.array(response.json()["data"]["ndarray"])

        results.append(
            {
                "puid": response.headers["seldon-puid"],
                "predictions": remote_predictions,
            }
        )

        if not np.allclose(local_predictions, remote_predictions):
            logging.warning("Deployed model returned different predictions than local model.")
            return False

    """
    Assert that seldon request logging works: Look into elasticsearch
    whether all requests made in this test have been logged correctly.
    """

    if elastic_search_index is not None:
        try:
            es_poll_interval = 5

            while elastic_search_logging_timeout > 0:
                n_logged_docs = len(get_docs_from_index(elastic_search_index)) - n_prior_docs_in_index
                logging.info(f"{n_logged_docs}/{n_batches * batch_size} documents have been logged into elasticsearch.")
                time.sleep(es_poll_interval)
                elastic_search_logging_timeout = elastic_search_logging_timeout - es_poll_interval
                if n_logged_docs == n_batches * batch_size:
                    break

            if n_logged_docs < n_batches * batch_size:
                logging.warning("Request logging is too slow.")
                return False

            assert len(results) == n_batches
            documents = get_docs_from_index(elastic_search_index)
            assert len(documents) >= n_batches * batch_size
            logging.info(f"Found {len(documents)} in the elasticsearch indes {elastic_search_index}.")

            for r in results:
                matched_documents = [doc for doc in documents if doc["_source"]["RequestId"] == r["puid"]]
                if not len(matched_documents) == batch_size:
                    logging.warning(
                        f"Would have expected {batch_size} documents in index with 'RequestId' {r['puid']}, found {len(matched_documents)}."
                    )
                    return False

                preds = np.array(r["predictions"])
                for doc in matched_documents:
                    logged_preds = np.array(doc["_source"]["response"]["payload"]["data"]["ndarray"])
                    if not np.allclose(preds, logged_preds):
                        logging.warning(
                            f"Logged predictions are {logged_preds} but should have been {preds} for puid {r['puid']}."
                        )
                        return False

            logging.info("All expected test predictions have been found in elasticsearch.")
        except Exception as e:
            logging.warning(f"Unexpected exception {e}")
            return False

    return True


def test_asynchronous_feedback(
    host: str,
    namespace: str,
    seldon_deployment_name: str,
    test_inputs: pd.core.frame.DataFrame,
    test_labels: pd.core.frame.DataFrame,
    request_timeout: int = 1,
    batch_size: int = 5,
    elastic_search_logging_timeout: int = 60,
    elastic_search_host: str = "http://elasticsearch-master.seldon-logs.svc.cluster.local:9200",
) -> bool:
    """
    Test that feedback can be sent asynchronously to the model and logged in elasticsearch.

    First, predictions are generated for a batch. The returned puid (prediction identifier)
    is used to then send ground truth feedback to the model for that prediction.

    It is asserted that the prediction request and the feedback are logged into elasticsearch
    in a specified amount of time.

    Args:
        host (str): host where the model is deployed. Can be localhost or the url
            of the istio ingressgateway.
        namespace (str): seldon deployment namespace
        seldon_deployment_name (str): seldon deployment name
        test_inputs (DataFrame): dataframe containing hold out inputs
        test_labels (DataFrame): dataframe containing corresponding hold out labels

    Kwargs:
        request_timeout (int): maximum duration (s) the feedback request can take
        batch_size (int): number of samples to generate predictions and send feedback for
        elastic_search_logging_timeout (int): maximum duration (s) the logging of the feedback
            into elasticsearch is allowed to take
        elastic_search_host (str): endpoint of elasticsearch

    Returns:
        A boolean indicating whether the test was successful.
    """
    import logging
    import numpy as np
    import requests
    import time
    from datetime import datetime

    pause_interval = 5

    prediction_endpoint = f"{host}/seldon/{namespace}/{seldon_deployment_name}/api/v1.0/predictions"
    feedback_endpoint = f"{host}/seldon/{namespace}/{seldon_deployment_name}/api/v1.0/feedback"
    elastic_search_index = f"inference-log-seldon-{namespace}-{seldon_deployment_name}-default"

    x, y = test_inputs.head(batch_size), test_labels.head(batch_size)
    y = np.eye(2, dtype=np.int64)[y]

    """
    Send prediction request to the model.
    """
    logging.info("Testing asynch feedback: send prediction request to the model.")
    data = {"data": {"names": test_inputs.columns.to_list(), "ndarray": x.values.tolist()}}

    response = requests.post(prediction_endpoint, json=data, timeout=request_timeout)

    if not response.ok:
        logging.warning("Prediction request not successful.")
        return False

    time.sleep(pause_interval)
    puid = response.headers["seldon-puid"]

    """
    Send feedback to the model.
    """
    time.sleep(pause_interval)

    logging.info("Testing asynch feedback: send feedback to the model.")
    feedback_tags = {"user": "model-validation", "date": f"{datetime.today().strftime('%d/%m/%Y')}"}
    for idx, label in enumerate(y):
        feedback_req = {
            "truth": {
                "data": {"names": ["malignant", "benign"], "ndarray": label.tolist()},
                "meta": {"tags": feedback_tags},
            },
        }
        response = requests.post(
            feedback_endpoint,
            json=feedback_req,
            headers={"seldon-puid": puid + f"-item-{idx}"},
            timeout=request_timeout,
        )

        if not response.ok:
            logging.warning("Asynchronous feedback not successful.")
            return False

    """Ensure that feedback is logged in elasticsearch."""
    es_poll_interval = 5

    while elastic_search_logging_timeout > 0:
        elastic_search_logging_timeout = elastic_search_logging_timeout - es_poll_interval
        time.sleep(es_poll_interval)

        docs = get_docs_from_index(elastic_search_index, host=elastic_search_host)
        try:
            if (
                n_logged_docs := len(
                    [d for d in docs if puid in d["_source"]["RequestId"] and "feedback" in d["_source"]]
                )
            ) == batch_size:
                logging.info("All feedback successfully logged in elasticsearch.")
                return True
            else:
                logging.info(f"{n_logged_docs}/{batch_size} feedback has been logged into elasticsearch.")
        except:
            logging.info("Feedback not logged into elasticsearch yet.")

    logging.warning("Feedback not logged into elasticsearch within specified timeout.")
    return False


def test_explainer(
    endpoint: str,
    test_df: pd.core.frame.DataFrame,
    timeout: int = 600,
) -> bool:
    """
    Test model explainer endpoint.

    Args:
        endpoint (str): uri of the deployed model explainer. Typically the service name
           of the istio ingressgateway with the path derived from the model's
           namespace and its name.
        test_df (DataFrame): pandas dataframe containing test data
    Kwargs:
        timeout (int): max time the explainer can take to become available

    Returns:
        A boolean indicating whether the test was successful
    """
    sample = test_df.head(1)
    request_body = {"data": {"names": ["image"], "ndarray": sample.values.tolist()}}

    poll_interval = 5
    while timeout > 0:
        response = requests.post(endpoint, json=request_body)
        if response.ok is True and response.json().get("data") is not None:
            logging.info("Explainer validated.")
            return True
        else:
            logging.info("Waiting for explainer to become available.")
        time.sleep(poll_interval)
        timeout = timeout - poll_interval

    logging.warning("Explainer did not become available in time.")
    return False
