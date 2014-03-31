#!/bin/bash

for image in venus cones teddy
do 
    for method in hocr reduction
    do 
        ~/Work/Vision/sum-of-submodular/release/stereo -m $method -i 100 --kappa 0.01 --alpha 10 --lambda 40000 $image
    done
    for method in spd-alpha spd-alpha-height 
    do
        for lb in 0 1
        do
            ~/Work/Vision/sum-of-submodular/release/stereo -m $method -i 100 --kappa 0.01 --alpha 10 --lambda 40000 --lower-bound $lb $image
        done
    done
done
