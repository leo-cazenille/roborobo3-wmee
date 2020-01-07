#!/bin/bash
set -e

uid=$1 #${1:-1000}
executionMode=$2 #${2:-"normal"}
nbRuns=$3
shift; shift; shift;

useradd -d /home/user -Ms /bin/bash -u $uid user
chown -R $uid /home/user

# Launch openssh server
/etc/init.d/ssh start

# Launch illumination
if [ "$executionMode" = "normal" ]; then
    if [ "$nbRuns" -eq "1" ]; then
        exec gosu user bash -c "cd /home/user/wmee/roborobo3; ln -s /home/user/results results; ./roborobo -l $@; rsync -avz /home/user/results/ /home/user/finalresults/"
    else
        exec gosu user bash -c "cd /home/user/wmee/roborobo3; ln -s /home/user/results results; for i in $(seq 1 $nbRuns | tr '\n' ' '); do sleep 1; ./roborobo $@ & sleep 1; done; wait; rsync -avz /home/user/results/ /home/user/finalresults/"
    fi
fi

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
