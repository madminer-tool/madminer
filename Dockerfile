# Docker image that contains madminer and root
FROM rootproject/root-ubuntu

USER root

RUN apt-get update && apt-get install -y \
    python-dev \
    python-tk \
    python3-tk \
    python3-pip

# Python2 pip is not longer shiped with Ubuntu (20.04+)
RUN curl "https://bootstrap.pypa.io/get-pip.py" --output get-pip.py && \
    python get-pip.py

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --upgrade --no-cache-dir madminer

WORKDIR /home/
