FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN git clone https://github.com/chandrikadeb7/Face-Mask-Detection.git

COPY test.py ./Face-Mask-Detection

WORKDIR Face-Mask-Detection

RUN pip3 install tensorflow==2.5.1
RUN pip3 install -r requirements.txt

RUN python3 test.py
