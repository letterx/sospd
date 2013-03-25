#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 learn-alg data-dir"
    exit 1
fi

for data_file in $2/data-small*.dat
do
    echo "***"
    echo "*** $(basename $data_file .dat) -- submodular"
    echo "***"
    mkdir -p submodular-results
    cd submodular-results
    time $1 -v 3 $data_file $2/$(basename $data_file .dat)-submodular.model $2/$(basename $data_file .dat)-submodular.predictions
    cd ..

    echo "***"
    echo "*** $(basename $data_file .dat) -- pairwise"
    echo "***"
    mkdir -p pairwise-results
    cd pairwise-results
    time $1 -v 3 --crf 1 $data_file $2/$(basename $data_file .dat)-pairwise.model $2/$(basename $data_file .dat)-pairwise.predictions
    cd ..

    echo "***"
    echo "*** $(basename $data_file .dat) -- contrast-pairwise"
    echo "***"
    mkdir -p gradient-results
    cd gradient-results
    time $1 -v 3 --crf 1 $data_file $2/$(basename $data_file .dat)-contrast.model $2/$(basename $data_file .dat)-gradient.predictions
    cd ..
done
