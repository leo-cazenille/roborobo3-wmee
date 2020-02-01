#!/bin/bash


declare -a configs

#configs+=("wmee-3201600-baseline 50")

#configs+=("wmee-3201600-WMAE4000x10x1e-4x30x10x1067200 50")
#configs+=("wmee-401600-baseline 20")
#configs+=("wmee-401600-WMAE4000x10x1e-4x30x10x134000 20")

#configs+=("wmee-401600-task1-baseline 20")
#configs+=("wmee-401600-task1-WMAE4000x10x1e-4x30x10x134000 20")

#configs+=("wmee-401600-task1_16sensors-baseline 10")
#configs+=("wmee-401600-task1_16sensors-WMAE4000x10x1e-4x30x10x134000 10")
#configs+=("wmee-401600-task1-WMAEMM4000x10x10x1e-4x1e-4x30x10x134000x268000 10")
#configs+=("wmee-401600-WMAEMM4000x10x10x1e-4x1e-4x30x10x134000x268000 2")
#configs+=("wmee-161600-WMAEMM4000x10x10x1e-4x1e-4x30x10x52800x105600 2")
#configs+=("wmee-161600-WMAEMM4000x10x10x1e-4x1e-3x30x10x52800x105600 2")
#configs+=("wmee-161600-WMAEMM4000x10x10x1e-4x1e-5x30x10x52800x105600 2")
configs+=("wmee-1601600-WMAEMM8000x10x10x1e-4x1e-4x30x10x268000x1072000 2")


printf -v allParams "%s\n" "${configs[@]}"
#echo $allParams | xargs --max-procs=$(nproc --all) -n 2 ./runDockerExpe.sh
echo $allParams | xargs --max-procs=12 -n 2 ./runDockerExpe.sh


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
