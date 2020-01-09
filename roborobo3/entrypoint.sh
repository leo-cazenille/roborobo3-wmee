#!/bin/bash
set -e

uid=$1 #${1:-1000}
executionMode=$2 #${2:-"normal"}
nbRuns=$3
configName=$4
shift; shift; shift; shift;

useradd -d /home/user -Ms /bin/bash -u $uid user
chown -R $uid /home/user

rm -fr /home/user/wmee/roborobo3/logs

# Launch openssh server
/etc/init.d/ssh start

# Launch illumination
if [ "$executionMode" = "normal" ]; then
    if [ "$nbRuns" -eq "1" ]; then
        exec gosu user bash -c "cd /home/user/wmee/roborobo3; mkdir -p /home/user/results/$configName; ln -s /home/user/results/$configName logs; SDL_VIDEODRIVER=offscreen ./roborobo -l config/$configName.properties; rsync -avz /home/user/results/ /home/user/finalresults/"
    else
        exec gosu user bash -c "cd /home/user/wmee/roborobo3; mkdir -p /home/user/results/$configName; ln -s /home/user/results/$configName logs; for i in $(seq 1 $nbRuns | tr '\n' ' '); do sleep 1; SDL_VIDEODRIVER=offscreen ./roborobo -l config/$configName.properties & sleep 1; done; wait; rsync -avz /home/user/results/ /home/user/finalresults/"
    fi
fi

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
