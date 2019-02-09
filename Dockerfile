#docker image that contains madminer and root
FROM rootproject/root-ubuntu16

USER root

RUN apt-get update && apt-get install -y python3-pip python python-pip python3-tk python-tk
RUN pip install --upgrade pip

RUN pip install --upgrade pip
RUN pip install madminer --upgrade  

WORKDIR /home/
