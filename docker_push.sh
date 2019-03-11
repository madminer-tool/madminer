#!/bin/bash
echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
export REPO=madminertool/docker-madminer
docker build -f Dockerfile -t $REPO .
docker push $REPO
