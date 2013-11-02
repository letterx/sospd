#!/bin/bash

for file in venus teddy cones
do
    for method in reduction-1 hocr-1 spd-alpha-0 spd-alpha-height-0
    do
        echo $file $method
        ~/Work/Vision/sum-of-submodular/experiments/stereo/disparity-diff $file-true/disp2.pgm results-ub/$file-$method.pgm
        tail -n 1 results-ub/$file-$method.stats
    done
    echo
    echo
done
