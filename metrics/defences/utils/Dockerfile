ARG METRIC_IMAGE=None
FROM ${METRIC_IMAGE} as base

#COPY --from=vg-code.gml-team.ru:4567/framework/attacks/datasets/coco:25e_shu /data /

RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install av
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install PyWavelets
RUN pip3 install IQA_pytorch
RUN pip3 install frozendict
RUN pip3 install lpips
RUN pip3 install torchmetrics
RUN pip3 install pytorch-wavelets
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install pyiqa


COPY ./read_dataset.py /read_dataset.py
COPY ./metrics.py /metrics.py
COPY ./bounds.json /bounds.json
COPY ./evaluate.py /evaluate.py
COPY ./read_dataset.py /read_dataset.py

COPY ./defence_evaluate.py /defence_evaluate.py
COPY ./defence_dataset.py /defence_dataset.py
COPY ./defence_scoring_methods.py /defence_scoring_methods.py
COPY ./defence_presets.json /defence_presets.json

COPY run.py /run.py
COPY ./dfsr[c] /dfsrc

COPY ./vqmt-14.1.12839.pro-Linux.deb /vqmt-14.1.12839.pro-Linux.deb
COPY ./config.json /vqmt_config.json
RUN apt install ocl-icd-libopencl1
RUN dpkg -i /vqmt-14.1.12839.pro-Linux.deb && vqmt -activate < /vqmt_config.json


