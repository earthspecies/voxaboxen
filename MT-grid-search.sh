#!/bin/sh

for detthresh in 0.55; do
#for detthresh in 0.4; do
    for combiouthresh in 0.5 0.55 0.6; do
    #for combiouthresh in 0.4; do
        for combdiscardthresh in  0.8 0.85 0.9; do
            combdiscardthresh2=$(echo ${combdiscardthresh}-0.075 | bc -l)
            (trap 'kill 0' SIGINT; python main.py train-model --project-config-fp=projects/MT_experiment/project_config.yaml --name=bidirectional-${detthresh}-${combiouthresh}-${combdiscardthresh} --lr=.00005 --batch-size=4 --n-epochs 20 --detection-threshold ${detthresh} --comb-iou-threshold ${combiouthresh} --comb-discard-threshold ${combdiscardthresh} & python main.py train-model --project-config-fp=projects/MT_experiment/project_config.yaml --name=bidirectional-${detthresh}-${combiouthresh}-${combdiscardthresh2} --lr=.00005 --batch-size=4 --n-epochs 20 --detection-threshold ${detthresh} --comb-iou-threshold ${combiouthresh} --comb-discard-threshold ${combdiscardthresh2} & wait) 
        done
    done
done
