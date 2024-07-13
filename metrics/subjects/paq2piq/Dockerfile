FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0 

COPY IQA-PyTorch /IQA-PyTorch
RUN cd /IQA-PyTorch \
 && pip install -r requirements.txt && python setup.py develop

COPY pyiqa/download.sh /
RUN bash download.sh

COPY paq2piq/src /src
COPY paq2piq/config.json /src

