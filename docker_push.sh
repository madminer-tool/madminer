#!/bin/bash
docker login -u "$DOCKER_USER" -p "$DOCKER_PASS"
export REPO=irinahub/docker-madminer
docker build -f Dockerfile -t $REPO .
docker push $REPO
