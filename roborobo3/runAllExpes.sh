#!/bin/bash


declare -a configs

# TODO
#configs+=("conf/rastrigin2vs6-noiseless-MutPolyBounded0.1x0.9-Grid64x64.yaml 2")


printf -v allParams "%s\n" "${configs[@]}"
#echo $allParams | xargs --max-procs=$(nproc --all) -n 2 ./runDockerExpe.sh
echo $allParams | xargs --max-procs=12 -n 2 ./runDockerExpe.sh


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
