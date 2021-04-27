# Docker image that contains madminer and root
FROM rootproject/root-ubuntu

USER root

RUN apt-get update && apt-get install -y \
    python3-tk \
    python3-pip

RUN pip3 install --upgrade --no-cache-dir pip && \
    pip3 install --upgrade --no-cache-dir madminer

WORKDIR /home/
