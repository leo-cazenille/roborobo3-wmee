FROM lcazenille/ubuntupyssh:latest
MAINTAINER leo.cazenille@gmail.com


RUN mkdir -p /home/user

# Create ssh key
RUN \
    mkdir /home/user/.ssh && \
    ssh-keygen -q -t rsa -N '' -f /home/user/.ssh/id_rsa && \
    cat /home/user/.ssh/id_rsa.pub > /home/user/.ssh/authorized_keys

# Download repositories
RUN \
    git clone https://github.com/leo-cazenille/roborobo3-wmee.git /home/user/wmee && \
    rm -fr /home/user/wmee/.git*

RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -yq wget unzip && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip -d /home/user/wmee/roborobo3/ libtorch-shared-with-deps-latest.zip && \
    rm -fr libtorch-shared-with-deps-latest.zip



#RUN pip3 install --upgrade --no-cache-dir -r /home/user/wmee/requirements.txt

# Prepare for entrypoint execution
CMD ["bash"]
#ENTRYPOINT ["/home/user/wmee/entrypoint.sh"]

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
