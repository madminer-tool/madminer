#!/bin/bash
docker login -u "$DOCKER_USER" -p "$DOCKER_PASS"
docker build -t madminertool/docker-madminer .
sudo docker push madminertool/docker-madminer
