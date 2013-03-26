#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 learn-alg data-dir c"
    exit 1
fi

common_args="-v 3 -l 1 -c $3 -w 2 --grabcut-unary 10"

for data_file in $2/data*-small.dat
do
    echo "***"
    echo "*** $(basename $data_file .dat) -- submodular"
    echo "***"
    time $1 $common_args --pairwise 0 --contrast-pairwise 0 --submodular 1 $data_file $2/$(basename $data_file -small.dat)-submodular.model

    echo "***"
    echo "*** $(basename $data_file .dat) -- pairwise"
    echo "***"
    time $1 $common_args --pairwise 1 --contrast-pairwise 0 --submodular 0 --crf ho $data_file $2/$(basename $data_file -small.dat)-pairwise.model

    echo "***"
    echo "*** $(basename $data_file .dat) -- contrast-pairwise"
    echo "***"
    time $1 $common_args --pairwise 1 --contrast-pairwise 1 --submodular 0 --crf ho $data_file $2/$(basename $data_file -small.dat)-contrast.model
done
