import logging
import pickle
from typing import Dict, Iterable

import numpy as np
from minio import Minio

logger = logging.getLogger("__model__")


class Classifier(object):
    """
    Deploy breast cancler classifier.

    The class has to have the same name as the python module file.

    Attributes:
        model: the model to deploy
    """

    def __init__(self, model_uri: str) -> None:
        """
        Construct Classifier instance.

        Arguments:
            model_uri (str): minio bucket uri to a sklearn model pickle
        """
        client = Minio("minio.flyte:9000", access_key="minio", secret_key="miniostorage", secure=False)
        bucket, fp = model_uri.split("/", 1)

        obj = client.get_object(
            bucket,
            fp,
        )

        self.model = pickle.load(obj)

        logger.info("Model has been loaded and initialized")

    def predict(self, X: np.array, names: Iterable[str], meta: Dict = None) -> np.array:
        """
        Predict method of Classifier.

        Args:
            X (np.array): numpy array containing features
            names (Iterable(str)): iterable of input (feature) names
            meta (Dict): dictionary containing meta data

        Returns:
            A numpy array containing model predictions
        """

        return self.model.predict_proba(X)

    def class_names(self) -> Iterable[str]:
        """
        Return the output class names of the model.
        Used by seldon-microservice to label the outputs.

        Returns:
            Iterable[str]: class names
        """
        return ["malignant", "benign"]
