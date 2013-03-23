#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: scale-data.sh scale input-dir output-dir"
    exit 1
fi

mkdir -p $3
mkdir -p $3/images/
mkdir -p $3/images-labels/
mkdir -p $3/images-gt/

echo $2/images
for file in $2/images/*.ppm
do
    pnmscale $1 $file > $3/images/$(basename $file .ppm).ppm
done
echo $2/images-labels
for file in $2/images-labels/*.ppm
do
    pnmscale $1 -nomix $file > $3/images-labels/$(basename $file .ppm).ppm
done
echo $2/images-gt
for file in $2/images-gt/*.ppm
do
    pnmscale $1 -nomix $file > $3/images-gt/$(basename $file .ppm).ppm
done
