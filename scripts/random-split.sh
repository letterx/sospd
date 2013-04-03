#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: $0 data-directory num-splits"
    exit 1
fi

(cd $1/images; ls *.ppm) | shuf > $1/random-order.dat

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
split_size=`expr $num_files / $num_splits`
echo "Split size: $split_size"

cat $1/header.txt > $1/data-all.dat
cat $1/random-order.dat >> $1/data-all.dat

for i in $(seq 1 $num_splits)
do
    small_file=$1/data-$i-$num_splits-small.dat
    large_file=$1/data-$i-$num_splits-large.dat

    cat $1/header.txt > $small_file
    cat $1/header.txt > $large_file

    start=$((( $i - 1 ) * $split_size))
    end=$(($i * $split_size))

    head -n $end $1/random-order.dat | tail -n $split_size >> $small_file
    head -n $start $1/random-order.dat >> $large_file
    tail -n $(($num_files - $end)) $1/random-order.dat >> $large_file

done

rm $1/header.txt
