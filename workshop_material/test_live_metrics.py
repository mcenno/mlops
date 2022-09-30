"""
Demonstrate live metrics shown in Grafana based on asynchronous feedback to a seldon model.

First, we log predictions and send correct feedback. The live metrics in grafana increase.
Then, we log predictions but provide random feedback to simulate decaying performance.

Note: the train-test split used in this demo is not guaranteed to be the same as the one
in the training pipeline. Just for demoing live-metrics.
"""
import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flytesnacks.seldon.model_validation import test_asynchronous_feedback

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="name of the seldon model")
args = parser.parse_args()

X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

test_asynchronous_feedback(
    host="http://localhost:8080",
    namespace="inference",
    seldon_deployment_name=args.model_name,
    test_inputs=X_test,
    test_labels=y_test,
    request_timeout=1,
    batch_size=20,
    elastic_search_logging_timeout=60,
    elastic_search_host="http://localhost:9200",
)

input("Press 'Return' to send random feedback continue.")

test_asynchronous_feedback(
    host="http://localhost:8080",
    namespace="inference",
    seldon_deployment_name=args.model_name,
    test_inputs=X_test,
    test_labels=y_test.sample(100),
    request_timeout=1,
    batch_size=20,
    elastic_search_logging_timeout=60,
    elastic_search_host="http://localhost:9200",
)
