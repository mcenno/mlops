FROM python:3.8

RUN pip install scikit-learn==1.0.1 seldon-core==1.5.0 minio boto3 awscli

WORKDIR ./deployment
ADD . .

ENTRYPOINT seldon-core-microservice Classifier --service-type MODEL --persistence 0