import logging

import pandas as pd

from minio import Minio

client = Minio("minio.flyte:9000", access_key="minio", secret_key="miniostorage", secure=False)


def save_df_in_bucket(df: pd.DataFrame, uri: str) -> None:
    """
    Save pandas dataframe as a csv in a bucket.

    Args:
        df (pd.DataFrame): data frame to be stored
        uri (str): uri to save the dataframe under.
            Assumption is that the first level is the bucket name, e.g.:
            "test_bucket/test_path/test_df.csv"
    """
    bucket, fp = uri.split("/", 1)

    from io import BytesIO

    csv = df.to_csv().encode("utf-8")
    client.put_object(bucket, fp, data=BytesIO(csv), length=len(csv), content_type="application/csv")
    logging.info(f"Saved dataframe under {uri}.")


def load_df_from_bucket(uri: str) -> pd.DataFrame:
    """
    Load a pandas dataframe from a bucket.

    Args:
        uri (str): uri to save the dataframe under.
            Assumption is that the first level is the bucket name, e.g.:
            "test_bucket/test_path/test_df.csv"
    """
    bucket, fp = uri.split("/", 1)
    obj = client.get_object(
        bucket,
        fp,
    )
    df = pd.read_csv(obj, index_col=0)
    logging.info(f"Loaded dataframe from {uri}.")
    return df


def download_file(uri: str, local_path: str) -> None:
    """
    Download a file from a bucket.

    Args:
        uri (str): uri to download the file from
        local_path (str): where the file is stored locally
    """
    bucket, fp = uri.split("/", 1)
    client.fget_object(bucket, fp, local_path)
    logging.info(f"Downloded file from {uri}.")


def upload_file(uri: str, local_path: str) -> None:
    """
    Save a local file in a bucket under the given uri.

    Args:
        uri (str): uri to save the file under.
        local_path (str): the local path of the file to upload
    """
    bucket, fp = uri.split("/", 1)
    client.fput_object(bucket, fp, local_path)
    logging.info(f"Uploaded file to {uri}.")
