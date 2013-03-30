#!/bin/bash

if [ $# -ne 4 ]
then
    echo "Usage: $0 learn-alg data-dir loss-scales cost-parameters"
    exit 1
fi

common_args="-v 3 -l 1 -w 2 --grabcut-unary 10"
data_file=$2/data-1-10-small.dat

for loss_scale in $3
do
    for c in $4
    do
        echo "***"
        echo "*** loss_scale = $loss_scale c = $c"
        echo "***"
        time $1 $common_args -c $c --loss-scale $loss_scale $data_file $2/loss-$loss_scale-c-$c.model
    done
done
