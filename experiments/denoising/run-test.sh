#!/bin/bash

for method in reduction reduction-grad
do
    echo $method
    echo
    ~/Work/Vision/sum-of-submodular/release/denoise -m $method medium-test
done

for method in spd-blur-random spd-grad 
do
    for lb in 0 1
    do
        echo $method $lb
        echo
        ~/Work/Vision/sum-of-submodular/release/denoise -m $method --lower-bound $lb medium-test
    done
done

