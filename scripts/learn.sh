#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 learn-alg data-dir c"
    exit 1
fi

for data_file in $2/data-small*.dat
do
    echo "***"
    echo "*** $(basename $data_file .dat)"
    echo "***"
    time $1 -v 3 -l 1 -c $3 -w 3 $data_file $2/$(basename $data_file .dat).model
done
