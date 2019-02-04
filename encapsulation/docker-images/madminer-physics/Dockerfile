FROM irinahub/docker-madminer-madgraph:latest

USER root 

RUN apt-get update && apt-get install -y  python python-pip  python-tk

RUN pip install --upgrade pip
RUN pip install madminer --upgrade 	&& pip install PyYAML && pip install 'numpy==1.15.4'  


WORKDIR /home/
#COPY /madminer ./madminer
COPY /code ./code

RUN chmod 755 -R ./code
#RUN chmod 755 -R ./madminer