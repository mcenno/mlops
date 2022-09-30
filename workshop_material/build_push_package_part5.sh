#!/bin/bash

REGISTRY="localhost:5000"
APP_NAME="workflow"
MODULE="flytesnacks.workflows"
VERSION=$(git show -s --format=%h)-${RANDOM}

TAG=$REGISTRY/${APP_NAME}:${VERSION}

docker build --tag ${TAG} .
docker push ${TAG}

echo "Docker image ${TAG} built."

TAG_EXPLAINER=$REGISTRY/${APP_NAME}-explainer:${VERSION}

docker build --tag ${TAG_EXPLAINER} -f Dockerfile.explainer .
docker push ${TAG_EXPLAINER}

echo "Docker image ${TAG_EXPLAINER} built."

pyflyte --pkgs ${MODULE} package --image k3d-registry.${TAG} --image explainer=k3d-registry.${TAG_EXPLAINER} -f

echo "Docker image ${TAG} and ${TAG_EXPLAINER} built and module ${MODULE} packaged with pyflyte"

flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version ${VERSION} --admin.endpoint localhost:30081 --admin.insecure --k8sServiceAccount model-deployer-service-account

DEPLOYMENT_NAME="classifier"
TAG_DEPLOYMENT=$REGISTRY/${DEPLOYMENT_NAME}:${VERSION}
docker build --tag ${TAG_DEPLOYMENT} flytesnacks/seldon/templates
docker push ${TAG_DEPLOYMENT}

echo "Pushed deployment image ${TAG_DEPLOYMENT}"
