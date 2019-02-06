FROM irinahub/docker-madminer:latest

USER root

RUN pip install madminer --upgrade  

WORKDIR /home/
