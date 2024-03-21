#!/bin/bash

CODE_DIR=`realpath $(dirname "$0")/..`

echo "Locate the project folder at ${CODE_DIR}"

cd ${CODE_DIR}

echo "test set"
JSON_DIR=${CODE_DIR}/KICDATA/predictseq
SPLIT=test

python -m diffab.KIC.kic \
        --json_dir $JSON_DIR \
        --split $SPLIT \
        --ncpu 16

echo "val set"
JSON_DIR=${CODE_DIR}/KICDATA/predictseq
SPLIT=val

python -m diffab.KIC.kic \
        --json_dir $JSON_DIR \
        --split $SPLIT \
        --ncpu 4

echo "train set"
JSON_DIR=${CODE_DIR}/KICDATA/predictseq
SPLIT=train

python -m diffab.KIC.kic \
        --json_dir $JSON_DIR \
        --split $SPLIT \
        --ncpu 100
