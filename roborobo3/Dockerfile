FROM lcazenille/ubuntupysshgcc:latest
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

# TODO caffe-cpu gcc-8 ? torch ? SDL dependencies
RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
#    apt-get install -yq wget unzip caffe-cpu libeigen3-dev libsdl2-2.0 libsdl2-dev libsdl2-image-dev libboost-all-dev automake && \
    apt-get install -yq wget unzip libeigen3-dev libboost-all-dev automake mercurial cmake libsdl2-dev libsdl2-image-dev && \
    rm -rf /var/lib/apt/lists/*

RUN \
    hg clone https://hg.libsdl.org/SDL && cd SDL && mkdir build && cd build && cmake -DVIDEO_OFFSCREEN=ON -DCMAKE_INSTALL_PREFIX=/usr .. && make -j 20 && make DESTDIR=/usr install && cd ../..

RUN \
    hg clone http://hg.libsdl.org/SDL_image && cd SDL_image && autoreconf -f -i; ./configure --prefix=/usr && make -j 20 && make install && cd ..

RUN \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.3.1%2Bcpu.zip && \
    unzip -d /home/user/wmee/roborobo3/ libtorch-cxx11-abi-shared-with-deps-1.3.1+cpu.zip && \
    rm -fr libtorch-cxx11-abi-shared-with-deps-1.3.1+cpu.zip


RUN \
    cd /home/user/wmee/roborobo3 && \
    make clean; make -j 24



#RUN pip3 install --upgrade --no-cache-dir -r /home/user/wmee/requirements.txt

# Prepare for entrypoint execution
#CMD ["bash"]
ENTRYPOINT ["/home/user/wmee/roborobo3/entrypoint.sh"]

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
