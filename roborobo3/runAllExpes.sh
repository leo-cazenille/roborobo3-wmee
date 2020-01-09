#!/bin/bash


declare -a configs

configs+=("wmee-100400-baseline 50")


printf -v allParams "%s\n" "${configs[@]}"
#echo $allParams | xargs --max-procs=$(nproc --all) -n 2 ./runDockerExpe.sh
echo $allParams | xargs --max-procs=12 -n 2 ./runDockerExpe.sh


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
