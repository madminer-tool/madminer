# Docker image that contains madminer and root
FROM rootproject/root:6.24.00-ubuntu20.04

USER root

RUN apt-get update && apt-get install -y \
    python3-tk \
    python3-pip

RUN pip3 install --upgrade --no-cache-dir pip && \
    pip3 install --upgrade --no-cache-dir madminer

WORKDIR /home/
