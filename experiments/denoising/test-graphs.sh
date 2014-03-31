#!/bin/bash

image=$1
sed "s/REPLACE_FILE/$image/" test-graph.gp > tmp.gp
gnuplot tmp.gp
rm -f tmp.gp
