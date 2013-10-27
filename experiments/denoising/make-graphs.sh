#!/bin/bash

for image in medium-test
do
    sed "s/REPLACE_FILE/$image/" energy-graph.gp > tmp.gp
    gnuplot tmp.gp
    rm -f tmp.gp
done
