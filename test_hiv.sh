#!/bin/bash

python design_pdb.py /datapool/data2/home/majianzhu/rectflow_seq_sabdab/hiv_target/chothia/3o2d-iMab_Repair_chothia.pdb \
                    --heavy H \
                    --light L \
                    -c ./configs/test/codesign_single_H3_hiv.yml \
                    --template ./hiv_target/template/template.pkl \
                    -b 32