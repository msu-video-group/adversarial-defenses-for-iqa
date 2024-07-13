#!/bin/bash

METRIC_NAME="${CI_JOB_NAME%:*}"

docker login -u gitlab-ci-token -p "$CI_JOB_TOKEN" "$CI_REGISTRY"
IMAGE="$CI_REGISTRY_IMAGE/metric/$METRIC_NAME:$LAUNCH_ID"
