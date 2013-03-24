#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 learn-alg data-dir c"
    exit 1
fi

for data_file in $2/data-small*.dat
do
    echo "***"
    echo "*** $(basename $data_file .dat) -- submodular"
    echo "***"
    time $1 -v 3 -l 1 -c $3 -w 3 --pairwise 0 --gradient 0 --submodular 1 $data_file $2/$(basename $data_file .dat)-submodular.model

    echo "***"
    echo "*** $(basename $data_file .dat) -- pairwise"
    echo "***"
    time $1 -v 3 -l 1 -c $3 -w 3 --pairwise 1 --gradient 1 --submodular 0 $data_file $2/$(basename $data_file .dat)-pairwise.model
done
