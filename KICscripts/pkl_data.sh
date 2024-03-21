#!/bin/bash

CODE_DIR=`realpath $(dirname "$0")/..`

echo "Locate the project folder at ${CODE_DIR}"

cd ${CODE_DIR}

echo "test set"
PDB_KIC_PATH=${CODE_DIR}/KICDATA/predictseq/test/pdb_kic
SPLIT=test
JSON_PATH=${CODE_DIR}/KICDATA/info

python -m diffab.KIC.pkl_data \
        --pdb_kic_path $PDB_KIC_PATH \
        --split $SPLIT \
        --json_path $JSON_PATH \
        --ncpu 30

echo "val set"
PDB_KIC_PATH=${CODE_DIR}/KICDATA/predictseq/val/pdb_kic
SPLIT=val
JSON_PATH=${CODE_DIR}/KICDATA/info

python -m diffab.KIC.pkl_data \
        --pdb_kic_path $PDB_KIC_PATH \
        --split $SPLIT \
        --json_path $JSON_PATH \
        --ncpu 60

echo "train set"
PDB_KIC_PATH=${CODE_DIR}/KICDATA/predictseq/train/pdb_kic
SPLIT=train
JSON_PATH=${CODE_DIR}/KICDATA/info

python -m diffab.KIC.pkl_data \
        --pdb_kic_path $PDB_KIC_PATH \
        --split $SPLIT \
        --json_path $JSON_PATH \
        --ncpu 100
