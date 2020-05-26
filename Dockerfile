#docker image that contains madminer and root
FROM rootproject/root-ubuntu16

USER root

RUN apt-get update && apt-get install -y \
    python
    python-tk
    python-pip
    python3-tk
    python3-pip


RUN pip install --upgrade --no-cache-dir pip && \
    pip install --upgrade --no-cache-dir madminer

WORKDIR /home/
