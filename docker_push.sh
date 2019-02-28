#!/bin/bash
docker login -u "$DOCKER_USER" -p "$DOCKER_PASS"
export REPO=madminertool/docker-madminer
docker build -f Dockerfile -t $REPO .
docker push $REPO
