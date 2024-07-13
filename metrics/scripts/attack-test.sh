#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh
. "$CI_PROJECT_DIR"/scripts/attack-generate-pipeline.sh

set -euxo pipefail
shopt -s extglob

trap
trap 'echo TRAPPED! "$@"' err

DATASETS_STORAGE="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/"

# Batch size for defences
BATCH_SIZE=16
# Debug dump frequency IN BATCHES. First image of the batch will be saved, if batch_idx % dump_freq == 0
DUMP_FREQ=25
# JSON file with presets for defences (Ignored if USE_DEFAULT_DEFENCE_PRESET=1)
PRESETS_JSON="./defence_presets.json"
# Path to source images for attacked dataset
SOURCE_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/koniq10k"
# CSV for MOSes for SOURCE_DATASET_PATH
MOSES_CSV_NAME="koniq10k_scores.csv"
# Preset for defence from defence_presets.json (Ignored if USE_DEFAULT_DEFENCE_PRESET=1)
DEFENCE_PRESET=4
# If set to 1, uses default initialization preset instead of DEFENCE_PRESET
USE_DEFAULT_DEFENCE_PRESET=0
# If set to 1, saves defended-attacked images to DEFENDED_DATASET_PATH
SAVE_DEFENDED_DATASET=1
# First image of  N-th BATCH is saved, -1 to save all images. I.e., saves first image of the batch if batch_idx % save_freq == 0
SAVE_FREQ=5

# On which attack type defence will be tested (no-defence -- no builtin defences in target metrics)
ATTACK_DEFENCE_TYPE="no-defence"
# On which attack preset defence will be tested (Not used, as now defences are tested on all attack presets at once)
# ATTACKS_PRESET=0
# Path to attacked dataset
ATTACKED_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/attacked-dataset/${ATTACK_DEFENCE_TYPE}/"

# specific save path if defences are using default preset.
if [[ "$USE_DEFAULT_DEFENCE_PRESET" == 1 ]]; then
  DEFENDED_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/defended-dataset/${METHOD_NAME}/${ATTACK_DEFENCE_TYPE}/default_preset/"
else
  DEFENDED_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/defended-dataset/${METHOD_NAME}/${ATTACK_DEFENCE_TYPE}/defence_preset_${DEFENCE_PRESET}/"
fi
# Amplitude is obsolete   
PRESET_AMPLITUDE=0.025

load_method_trainable_blackbox "$METHOD_NAME"
load_method_non_uap_blackbox "$METHOD_NAME"

if [[ "$METHOD_TRAINABLE_BLACKBOX" == 1 ]]; then
 
	TRAIN_DATASETS=( BLACK-BOX )
	TRAIN_DATASET_PATHS=( 
		"/train/black-box-dataset"
	)
else
	TRAIN_DATASETS=( COCO VOC2012 )
	TRAIN_DATASET_PATHS=( 
		"/train/COCO_25e_shu/train"
		"/train/VOC2012/JPEGImages"
	)
fi

if [[ "$NON_UAP_BLACKBOX_METHODS" == 1 ]]; then
 
	TEST_DATASETS=( SMALL-KONIQ-50 )
	TEST_DATASET_PATHS=(
		#"/test/black-box-dataset"
    "/test/quality-sampled-datasets/koniq_sampled_MOS/50_10_clusters"
	)
else
	TEST_DATASETS=( KONIQ-1000 )
  TEST_DATASET_PATHS=( 
  #  "/test/DERF"
  #  "/test/small-sampled-datasets/nips_sampled/10"
    "/test/quality-sampled-datasets/koniq_sampled_MOS/1000_10_clusters"
  )
fi
 

load_metric_launch_params "$METRIC_NAME" "$METHOD_NAME"
load_method_trainable "$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"
load_video_metric "$METRIC_NAME"

if [[ "$VIDEO_METRIC" == 1 ]]; then
    video_param="--video-metric"
#    TEST_DATASETS=( DERF )
#    TEST_DATASET_PATHS=( 
#        "/test/DERF"
#     )
else
    video_param=""
fi




is_fr=($(jq -r '.is_fr' "${CI_PROJECT_DIR}/subjects/${METRIC_NAME}/config.json"))

if [[ "$is_fr" == true ]]; then
    quality_param="--jpeg-quality 80"
else
    quality_param=""
fi

if [[ "$METHOD_NAME" == "noattack" ]]; then
    if [[ "$is_fr" == true ]]; then
        TEST_DATASETS=( NIPS_200 )
        TEST_DATASET_PATHS=( 
            "/test/NIPS 2017/images_200"
         )
    else
        TEST_DATASETS=( DIV2K_valid_HR )
        TEST_DATASET_PATHS=( 
            "/test/DIV2K_valid_HR"
         )
    fi
fi



codecs_param="--codec libx264 libx265"



cd "defences/$METHOD_NAME"
cp -a run.py "$CACHE/"


DUMPS_STORAGE="${CACHE}/dumps"
mkdir -p DUMPS_STORAGE


DOCKER_PARAMS=( --init --gpus device="${CUDA_VISIBLE_DEVICES-0}" -t --rm --name "gitlab-$CI_PROJECT_PATH_SLUG-$CI_JOB_ID" )
if [[ $GML_SHARED == *"maindev"* ]]; then
  DOCKER_PARAMS+=("--add-host=titan.gml-team.ru:10.32.0.32")
fi

  
docker run "${DOCKER_PARAMS[@]}" \
  -v "$DATASETS_STORAGE":"/test":ro \
  -v "$CACHE:/artifacts" \
  -v "$CACHE/run.py:/run.py" \
  -v "$DUMPS_STORAGE":"/dumps" \
  -v "$ATTACKED_DATASET_PATH":"/attacked-dataset" \
  -v "$SOURCE_DATASET_PATH":"/source-dataset" \
  -v "$DEFENDED_DATASET_PATH":"/defended-dataset" \
  "$IMAGE" \
  python ./run.py \
    "${METRIC_LAUNCH_PARAMS[@]}" \
    --metric  "${METRIC_NAME}" \
    --device "cuda:0" \
    --defence "${METHOD_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --save-defended "${SAVE_DEFENDED_DATASET}" \
    --attacked-dataset-path "/attacked-dataset" \
    --src-dir "/source-dataset/512x384" \
    --mos-path "/source-dataset/${MOSES_CSV_NAME}" \
    --save-path "/artifacts/${METRIC_NAME}_test.csv" \
    --defended-dataset-path "/defended-dataset" \
    --dump-path "/dumps" \
    --log-file "/artifacts/log.csv" \
    --defence-preset "${DEFENCE_PRESET}" \
    --use-default-preset "${USE_DEFAULT_DEFENCE_PRESET}" \
    --presets-json-path "${PRESETS_JSON}" \
    --dump-freq "${DUMP_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
  | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
  
#mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"

zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
#mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"

