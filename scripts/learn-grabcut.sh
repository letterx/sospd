#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 learn-alg data-dir c"
    exit 1
fi

data_file=$2/data-1-5-small.dat
for grabcut_level in 0 1 5 10
do
    echo "***"
    echo "*** Grabcut level: $grabcut_level"
    echo "***"
    time $1 -v 3 -l 1 -c $3 -w 2 --pairwise 0 --contrast-pairwise 0 --submodular 1 --grabcut-unary $grabcut_level $data_file $2/$(basename $data_file -small.dat)-grabcut-$grabcut_level.model
done
