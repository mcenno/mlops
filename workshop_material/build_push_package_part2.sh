#!/bin/bash

REGISTRY="localhost:5000"
APP_NAME="workflow"
MODULE="flytesnacks.workflows"
VERSION=$(git show -s --format=%h)-${RANDOM}

TAG=$REGISTRY/${APP_NAME}:${VERSION}

docker build --tag ${TAG} .
docker push ${TAG}

pyflyte --pkgs ${MODULE} package --image k3d-registry.${TAG} -f

echo "Docker image ${TAG} built and module ${MODULE} packaged with pyflyte"

flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version ${VERSION} --admin.endpoint localhost:30081 --admin.insecure --k8sServiceAccount model-deployer-service-account

DEPLOYMENT_NAME="classifier"
TAG_DEPLOYMENT=$REGISTRY/${DEPLOYMENT_NAME}:${VERSION}
docker build --tag ${TAG_DEPLOYMENT} flytesnacks/seldon/templates
docker push ${TAG_DEPLOYMENT}

echo "Pushed deployment image ${TAG_DEPLOYMENT}"
