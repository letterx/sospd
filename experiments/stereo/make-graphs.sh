#!/bin/bash

for image in venus teddy cones
do
    sed "s/REPLACE_FILE/$image/" energy-graph.gp > tmp.gp
    gnuplot tmp.gp
    rm -f tmp.gp
done
