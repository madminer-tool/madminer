FROM python:3.6

RUN apt-get update && apt-get -y install vim

COPY requirements.txt requirements.txt

# pip install 
RUN apt-get -y install python3-pip && pip install --no-cache -r requirements.txt 

WORKDIR /home

COPY code/ ./code
