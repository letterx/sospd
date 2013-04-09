#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 data-directory num-splits"
    exit 1
fi

(ls $1/images/*.ppm) | shuf > $1/random-order.dat

num_files=`cat $1/random-order.dat | wc -l`
echo "Num files: $num_files"

if [ $num_files -eq 0 ]
then
    echo "Error: no .ppm files in $1/images"
    exit 1
fi

echo "# The images directory" > $1/header.txt
echo $1/images/ >> $1/header.txt
echo "# The trimap directory" >> $1/header.txt
echo $1/images-labels/ >> $1/header.txt
echo "# The ground-truth directory" >> $1/header.txt
echo $1/images-gt/ >> $1/header.txt
echo "# The files" >> $1/header.txt

num_splits=$2

cat $1/header.txt > $1/data-all.dat
cat $1/random-order.dat >> $1/data-all.dat

for i in $(seq 1 $num_splits)
do
    (ls $1/images/*.ppm) | shuf > $1/random-order.dat

    train_file=$1/train-$i.dat
    test_file=$1/test-$i.dat
    validate_file=$1/validate-$i.dat

    cat $1/header.txt > $train_file
    cat $1/header.txt > $test_file
    cat $1/header.txt > $validate_file

    head -n 75 $1/random-order.dat >> $train_file
    tail -n 76 $1/random-order.dat | head -n 38 >> $validate_file
    tail -n 38 $1/random-order.dat >> $test_file
done

rm $1/header.txt
