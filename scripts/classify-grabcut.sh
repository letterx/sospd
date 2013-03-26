#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 learn-alg data-dir"
    exit 1
fi

data_file=$2/data-1-5-large.dat
for grabcut_level in 0 1 5 10
do
    echo "***"
    echo "*** Grabcut level: $grabcut_level"
    echo "***"
    mkdir -p $2/grabcut-$grabcut_level-results
    time $1 -v 3 --output-dir $2/grabcut-$grabcut_level-results/ $data_file $2/$(basename $data_file -large.dat)-grabcut-$grabcut_level.model /dev/null
done
