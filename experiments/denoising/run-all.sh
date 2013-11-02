#!/bin/bash

for file in ./test/*.pgm
do
    basename=$(basename $file .pgm)
    for method in reduction reduction-alpha reduction-grad
    do
        ~/Work/Vision/sum-of-submodular/experiments/denoising/denoise -m $method --time 10 ./test-noisy/$basename
    done

    for method in spd-alpha spd-alpha-height spd-blur-random spd-grad 
    do
        for lb in 0 1
        do
            ~/Work/Vision/sum-of-submodular/experiments/denoising/denoise -m $method --time 10 --lower-bound $lb ./test-noisy/$basename
        done
    done
done

