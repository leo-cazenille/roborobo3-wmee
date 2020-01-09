#!/bin/bash

configFile=${1:-}
nbRuns=${2:-1}
executionMode=${3:-"normal"}
memoryLimit=32G
#resultsPath=$(pwd)/results
resultsPathInContainer=/home/user/results
finalresultsPath=$(pwd)/results
finalresultsPathInContainer=/home/user/finalresults
imageName=${4:-"localhost:5000/wmee"}
uid=$(id -u)
confPath=$(pwd)/config
confPathInContainer=/home/user/wmee/roborobo3/config
priorityParam="-c 128"

if [ ! -d $finalresultsPath ]; then
    mkdir -p $finalresultsPath
fi

inDockerGroup=`id -Gn | grep docker`
if [ -z "$inDockerGroup" ]; then
    sudoCMD="sudo"
else
    sudoCMD=""
fi
dockerCMD="$sudoCMD docker"

if [ -d "$confPath" ]; then
    confVolParam="-v $confPath:$confPathInContainer"
else
    confVolParam=""
fi

if [ "$executionMode" = "normal" ]; then
    #exec $dockerCMD run -i -m $memoryLimit --rm $priorityParam -v $resultsPath:$resultsPathInContainer $confVolParam $imageName  "$uid" "normal" "-c $configFile --runs $nbRuns"
    exec $dockerCMD run -i -m $memoryLimit --rm $priorityParam --mount type=tmpfs,tmpfs-size=8589934592,target=$resultsPathInContainer --mount type=bind,source=$finalresultsPath,target=$finalresultsPathInContainer  $confVolParam $imageName  "$uid" "normal" "$nbRuns" "$configFile"
fi

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
