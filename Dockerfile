# This the base image for running FEDOT in container
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/FEDOT

COPY requirements.txt /home/FEDOT/requirements.txt
RUN pip3 install pip==19.3.1 && \
    pip install --trusted-host pypi.python.org -r /home/FEDOT/requirements.txt

COPY . /home/FEDOT

ENV PYTHONPATH /home/FEDOT