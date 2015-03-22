#!/bin/bash

application="cell-tracking"
ogm="/Users/afix/Work/Vision/opengm/build/src/interfaces/commandline/double/opengm_min_sum"
data_dir="/Users/afix/Work/Vision/opengm-benchmarks/$application"
data_files="ogm_model.h5"
timestamp=$(date +%y-%m-%d-%H-%M-%S)
output_dir="/Users/afix/Work/Vision/CVPR15-results/all-results/$application/$timestamp"
output_link="/Users/afix/Work/Vision/CVPR15-results/$application-results"

common_args="-p 1"

sos_ub_pairwise="-a SOS_UB --ubType pairwise"
sos_ub_chen="-a SOS_UB --ubType chen"
sos_ub_gurobiL1="-a SOS_UB --ubType gurobiL1"
sos_ub_gurobiL2="-a SOS_UB --ubType gurobiL2"
sos_ub_gurobiLInfty="-a SOS_UB --ubType gurobiLInfty"
icm="-a ICM"
bp="-a BELIEFPROPAGATION"
trbp="-a TRBP"
lazy_flipper="-a LAZYFLIPPER"
inf_and_flip="-a INFANDFLIP"
alpha_fusion="-a ALPHAEXPANSIONFUSION"

methods="sos_ub_pairwise sos_ub_chen sos_ub_gurobiL1 sos_ub_gurobiL2 sos_ub_gurobiLInfty icm bp trbp lazy_flipper inf_and_flip alpha_fusion"

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
        $ogm -m $data_dir/$file $common_args $args -o $output_dir/$output >> $output_dir/output.txt
    done
done

