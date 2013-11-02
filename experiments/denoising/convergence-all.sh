#!/bin/bash

for method in reduction-1 reduction-grad-1 hocr-grad-1 spd-alpha-1 spd-blur-random-1 spd-grad-1 spd-alpha-0 spd-blur-random-0 spd-grad-0
do
    echo $method
    ./convergence-stats.py results-with-ub/*$method.stats
    echo
done
