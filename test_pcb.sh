#!/bin/bash
python test_ssd.py
cd /opt/Data/tsl/DeepPCB/ssd-PCB-GPP-MaxPooling/results/
zip gt.zip gt_*.txt
zip res.zip res_*.txt
cd ../../evaluation
cp ../ssd-PCB-GPP-MaxPooling/results/gt.zip ./
cp ../ssd-PCB-GPP-MaxPooling/results/res.zip ./
python script.py -g=gt.zip -s=res.zip