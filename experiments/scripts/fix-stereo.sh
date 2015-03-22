#!/bin/bash

for image in cones teddy venus
do 
    for method in sospd_tripleInfty sospd_triple sospd_pairwise sospd_cvpr14 sospd_cardinality reduction_fusion
    do 
        echo ${image}_$method
        ~/Work/Vision/sum-of-submodular/release/opengm-to-stereo --image ~/Work/Vision/Higher-order-applications/stereo/${image} --ogmFile stereo-results/${image}_$method.txt --output stereo-results/${image}_$method.png
    done
done

rm -f disp-diff.txt
for image in cones teddy venus
do 
    for method in sospd_tripleInfty sospd_triple sospd_pairwise sospd_cvpr14 sospd_cardinality reduction_fusion
    do 
        echo ${image}_$method >> disp-diff.txt
        ~/Work/Vision/sum-of-submodular/release/disparity-diff ~/Work/Vision/Archived-Results/CVPR14-results/stereo/${image}-true/disp2.pgm stereo-results/${image}_${method}.png >> disp-diff.txt
    done
done
