#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh

set -euxo pipefail

cd "defences/$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"

load_cadv "$METHOD_NAME"

load_diffpure "$METHOD_NAME"

load_disco "$METHOD_NAME"

load_method_trainable "$METHOD_NAME"

# to work on LOMONOSOV-270. user must be maindev 
DOCKER_PARAMS=( )
if [[ $GML_SHARED == *"maindev"* ]]; then
  DOCKER_PARAMS=( --add-host=titan.gml-team.ru:10.32.0.32 )
fi

cp -a "$CI_PROJECT_DIR"/defences/utils/. ./
cp -a "${GML_SHARED}/vqmt/." ./

cat Dockerfile

if (( METHOD_DISCO != 0 )); then
    printf "\nRUN pip3 install geotorch\n">> Dockerfile
    printf "\nRUN pip3 install torchdiffeq\n">> Dockerfile
    printf "\nRUN pip3 install tensorboardX\n">> Dockerfile
fi

if (( METHOD_DIFFPURE != 0 )); then 

    printf "\nRUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python \
        python3-dev \
        python3-pip \
        python3-setuptools \
        zlib1g-dev \
        swig \
        cmake \
        vim \
        locales \
        locales-all \
        screen \
        zip \
        unzip\n">> Dockerfile
    printf "\nRUN apt-get clean\n">> Dockerfile

    printf "\nRUN pip3 install numpy==1.19.4 \
                pyyaml==5.3.1 \
                wheel==0.34.2 \
                pillow==7.2.0 \
                matplotlib==3.3.0 \
                tqdm==4.56.1 \
                tensorboardX==2.0 \
                seaborn==0.10.1 \
                pandas \
		        opencv-python \
                requests==2.25.0 \
                xvfbwrapper==0.2.9 \
                torchdiffeq==0.2.1 \                
                scikit-image==0.19 \
                timm \
                lmdb \
                Ninja \
                foolbox \
                torchsde \
                git+https://github.com/RobustBench/robustbench.git@v1.0\n">> Dockerfile
    
    printf "\nRUN pip3 install scipy==1.10\n">> Dockerfile
fi

if (( METHOD_TRAINABLE != 0 )); then
    printf "\nCOPY train.py /train.py\n" >> Dockerfile
fi

if (( METHOD_CADV != 0)); then
    printf "\nRUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth  https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth \
 && rm cadv-colorization-model.pth.1\n" >> Dockerfile
fi

if (( METHOD_MULTIMETRIC != 0 )); then 
    for i in "${METRICS[@]}"
    do
    	printf "\nCOPY --from=${CI_REGISTRY_IMAGE}/metric/${i}:${LAUNCH_ID} /src /${i}/\n" >> Dockerfile
    	weights=($(jq -r '.weight' "${CI_PROJECT_DIR}/subjects/${i}/config.json"  | tr -d '[]," '))
    	for fn in "${weights[@]}"
        do
            printf "\nCOPY --from=${CI_REGISTRY_IMAGE}/metric/${i}:${LAUNCH_ID} /${fn} /${i}/${fn}\n" >> Dockerfile
        done
    done
    printf "\nCOPY run.py /run.py\n" >> Dockerfile
    docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" .
    
else
    docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" --build-arg METRIC_IMAGE="$CI_REGISTRY_IMAGE/metric/$METRIC_NAME:$LAUNCH_ID" .
fi
docker push "$IMAGE"
