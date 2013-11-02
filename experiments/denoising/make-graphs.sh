#!/bin/bash

image=$1
sed "s/REPLACE_FILE/$image/" energy-graph.gp > tmp.gp
gnuplot tmp.gp
rm -f tmp.gp
