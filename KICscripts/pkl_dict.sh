#!/bin/bash

CODE_DIR=`realpath $(dirname "$0")/..`

echo "Locate the project folder at ${CODE_DIR}"

cd ${CODE_DIR}

INPATH=${CODE_DIR}/KICDATA/predictseq/
OUTPATH=${INPATH}/dict.pkl

python -m diffab.KIC.pkl_dict \
        --path $INPATH \
        --outpath $OUTPATH \
        --test \
        --val \
        --train