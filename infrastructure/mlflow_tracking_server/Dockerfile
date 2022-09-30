FROM python:3.8-slim

RUN pip install --upgrade pip && \
    pip install psycopg2-binary && \
    pip install mlflow && \
    pip install boto3 && \
    pip install awscli

CMD ["sh", "-c", "mlflow server --backend-store-uri $FILE_STORE --default-artifact-root $ARTIFACT_STORE --host 0.0.0.0 --port 5000"]
