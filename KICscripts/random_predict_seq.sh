#!/bin/bash

CODE_DIR=`realpath $(dirname "$0")/..`

echo "Locate the project folder at ${CODE_DIR}"

cd ${CODE_DIR}

echo "test set"

JSON_PATH=${CODE_DIR}/KICDATA/info/test.json
SAVE_PATH=${CODE_DIR}/KICDATA/predictseq

python -m diffab.KIC.random_predict_seq \
        --json_path $JSON_PATH \
        --save_path $SAVE_PATH

echo "val set"

JSON_PATH=${CODE_DIR}/KICDATA/info/val.json
SAVE_PATH=${CODE_DIR}/KICDATA/predictseq

python -m diffab.KIC.random_predict_seq \
        --json_path $JSON_PATH \
        --save_path $SAVE_PATH

echo "train set"

JSON_PATH=${CODE_DIR}/KICDATA/info/train.json
SAVE_PATH=${CODE_DIR}/KICDATA/predictseq

python -m diffab.KIC.random_predict_seq \
        --json_path $JSON_PATH \
        --save_path $SAVE_PATH
