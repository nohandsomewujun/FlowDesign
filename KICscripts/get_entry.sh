#!/bin/bash

CODE_DIR=`realpath $(dirname "$0")/..`

echo "Locate the project folder at ${CODE_DIR}"

cd ${CODE_DIR}

echo "test set"

SPLIT=test
SAVE_DIR=${CODE_DIR}/KICDATA/info/
echo "save in ${SAVE_DIR}"
#python -m diffab.KIC.get_entry \
        --split $SPLIT \
        --save_dir $SAVE_DIR

echo "val set"

SPLIT=val
echo "save in ${SAVE_DIR}"
#python -m diffab.KIC.get_entry \
        --split $SPLIT \
        --save_dir $SAVE_DIR

echo "train set"

SPLIT=train
SAVE_DIR=${CODE_DIR}/KICDATA/info/
python -m diffab.KIC.get_entry \
        --split $SPLIT \
        --save_dir $SAVE_DIR
