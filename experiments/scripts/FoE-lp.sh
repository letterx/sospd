#!/bin/bash

application="FoE"
ogm="/Users/afix/Work/Vision/opengm/build/src/interfaces/commandline/double/opengm_min_sum"
data_dir="/Users/afix/Work/Vision/opengm-benchmarks/$application"
#data_files="$data_dir/*.h5"
data_files="$data_dir/101085.h5 $data_dir/101087.h5 $data_dir/102061.h5 $data_dir/103070.h5 $data_dir/105025.h5 $data_dir/106024.h5 $data_dir/108005.h5 $data_dir/108070.h5 $data_dir/108082.h5 $data_dir/109053.h5"
timestamp=$(date +%y-%m-%d-%H-%M-%S)
output_dir="/Users/afix/Work/Vision/CVPR15-results/all-results/$application/$timestamp"
output_link="/Users/afix/Work/Vision/CVPR15-results/$application-lp-results"

maxIt=300
timeout=7200
sospd_pairwise="    -a SOSPD --ubType pairwise     --maxIt $maxIt --proposal blur"
sospd_chen="        -a SOSPD --ubType chen         --maxIt $maxIt --proposal blur"
sospd_cardinality=" -a SOSPD --ubType cardinality  --maxIt $maxIt --proposal blur"
sospd_gurobiL1="    -a SOSPD --ubType gurobiL1     --maxIt $maxIt --proposal blur"
sospd_gurobiL2="    -a SOSPD --ubType gurobiL2     --maxIt $maxIt --proposal blur"
sospd_gurobiLInfty="-a SOSPD --ubType gurobiLInfty --maxIt $maxIt --proposal blur"
sospd_cvpr14="      -a SOSPD --ubType cvpr14       --maxIt $maxIt --proposal blur"
trws="-a TRWS"
icm="-a ICM"
bp="-a BELIEFPROPAGATION"
trbp="-a TRBP"
lazy_flipper="-a LAZYFLIPPER"
inf_and_flip="-a INFANDFLIP"
reduction_fusion="      -a Fusion -f QPBO -g BLUR --numIt $maxIt"
sos_pair_fusion="       -a Fusion -f SOS  -g BLUR --numIt $maxIt --ubType pairwise"
sos_cvpr_fusion="       -a Fusion -f SOS  -g BLUR --numIt $maxIt --ubType cvpr14"
sos_chen_fusion="       -a Fusion -f SOS  -g BLUR --numIt $maxIt --ubType chen"
sos_cardinality_fusion="-a Fusion -f SOS  -g BLUR --numIt $maxIt --ubType cardinality"

sospd_pairwise_grad="    -a SOSPD --ubType pairwise           --maxIt $maxIt --proposal grad"
sospd_chen_grad="        -a SOSPD --ubType chen               --maxIt $maxIt --proposal grad"
sospd_cardinality_grad=" -a SOSPD --ubType cardinality        --maxIt $maxIt --proposal grad"
sospd_gurobiL1_grad="    -a SOSPD --ubType gurobiL1           --maxIt $maxIt --proposal grad"
sospd_gurobiL2_grad="    -a SOSPD --ubType gurobiL2           --maxIt $maxIt --proposal grad"
sospd_gurobiLInfty_grad="-a SOSPD --ubType gurobiLInfty       --maxIt $maxIt --proposal grad"
sospd_cvpr14_grad="      -a SOSPD --ubType cvpr14             --maxIt $maxIt --proposal grad"
sospd_latPair_grad="     -a SOSPD --ubType latticePairwise    --maxIt $maxIt --proposal grad"
sospd_latCard_grad="     -a SOSPD --ubType latticeCardinality --maxIt $maxIt --proposal grad"
reduction_fusion_grad="      -a Fusion -f QPBO -g GRAD --numIt $maxIt"
sos_pair_fusion_grad="       -a Fusion -f SOS  -g GRAD --numIt $maxIt --ubType pairwise"
sos_cvpr_fusion_grad="       -a Fusion -f SOS  -g GRAD --numIt $maxIt --ubType cvpr14"
sos_chen_fusion_grad="       -a Fusion -f SOS  -g GRAD --numIt $maxIt --ubType chen"
sos_cardinality_fusion_grad="-a Fusion -f SOS  -g GRAD --numIt $maxIt --ubType cardinality"

#methods="sospd_pairwise sospd_cvpr14 sos_pair_fusion sos_cvpr_fusion"
methods="\
         sospd_pairwise_grad \
         sospd_cvpr14_grad \
         sospd_cardinality_grad \
         sospd_gurobiL1_grad \
         sospd_gurobiLInfty_grad"


common_args="-p 1 --timeout $timeout"

mkdir -p $output_dir
rm -f $output_link
ln -s $output_dir $output_link

cp $0 $output_dir/

for file in $data_files
do
    echo $(basename $file .h5) >> $output_dir/files.txt
done

for method in $methods
do
    echo $method >> $output_dir/methods.txt
done

for file in $data_files
do
    for method in $methods
    do
        echo "*** Optimizing $file $method ***"
        output=$(basename $file .h5)_$method.txt
        eval args=\$$method
        $ogm -m $file $common_args $args -o $output_dir/$output >> $output_dir/output.txt
    done
done

