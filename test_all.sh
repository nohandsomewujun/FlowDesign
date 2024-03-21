#!/bin/bash

template_dict=./template/dict.pkl

for i in {0..19}
do
   python design_testset.py --template_dict $template_dict -c ./configs/test/codesign_single_H3.yml -b 32 $i
done