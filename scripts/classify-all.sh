#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 learn-alg data-dir"
    exit 1
fi

for data_file in $2/data*-large.dat
do
    echo "***"
    echo "*** $(basename $data_file .dat) -- submodular"
    echo "***"
    mkdir -p $2/submodular-results
    time $1 -v 3 --output-dir $2/submodular-results/ $data_file $2/$(basename $data_file -large.dat)-submodular.model /dev/null

    echo "***"
    echo "*** $(basename $data_file .dat) -- pairwise"
    echo "***"
    mkdir -p $2/pairwise-results
    time $1 -v 3 --output-dir $2/pairwise-results/ --crf ho $data_file $2/$(basename $data_file -large.dat)-pairwise.model /dev/null

    echo "***"
    echo "*** $(basename $data_file .dat) -- contrast-pairwise"
    echo "***"
    mkdir -p contrast-results
    time $1 -v 3 --output-dir $2/contrast-results/ --crf ho $data_file $2/$(basename $data_file -large.dat)-contrast.model /dev/null
done
