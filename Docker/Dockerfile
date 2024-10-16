# FROM python:3.10-bullseye
# FROM ubuntu:latest 
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
# ===========================================================
# Install apt dependencies
# -----------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get clean
RUN apt-get update
RUN apt-get install -y \
    nano \
    libgl1 \
    python3.10 \
    python3-pip
RUN cd /home

RUN rm -rf /var/lib/apt/lists/*


COPY ./ ./home/

RUN cd /home && \
    pip install . 


RUN mkdir -p /pruner/dataset_yaml \
    mkdir -p /pruner/output \
    mkdir -p /pruner/models 

WORKDIR /home

CMD [ "/bin/bash" ]
