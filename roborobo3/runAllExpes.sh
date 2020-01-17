#!/bin/bash


declare -a configs

#configs+=("wmee-3201600-baseline 50")

#configs+=("wmee-3201600-WMAE4000x10x1e-4x30x10x1067200 50")
#configs+=("wmee-401600-baseline 20")
#configs+=("wmee-401600-WMAE4000x10x1e-4x30x10x134000 20")

configs+=("wmee-401600-task1-baseline 20")
configs+=("wmee-401600-task1-WMAE4000x10x1e-4x30x10x134000 20")

printf -v allParams "%s\n" "${configs[@]}"
#echo $allParams | xargs --max-procs=$(nproc --all) -n 2 ./runDockerExpe.sh
echo $allParams | xargs --max-procs=12 -n 2 ./runDockerExpe.sh


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
