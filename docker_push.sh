#!/bin/bash
docker login -u "$DOCKERUSER" -p "$DOCKERPASS"
docker build -t madminertool/docker-madminer .
docker push madminertool/docker-madminer
