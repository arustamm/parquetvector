FROM ubuntu:22.04
LABEL maintainer="Rustam Akhmadiev"  
LABEL email="akhmarust@sep.stanford.edu"

# need to set the timezone when building
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y tzdata

RUN apt-get -y update &&\
    apt-get -y  install g++ python3-numpy git make gcc libboost-all-dev  libboost-dev &&\
    apt-get -y install python3-dev python3-pytest python3-venv libtbb-dev&& \
    apt-get -y  install gfortran libfftw3-3 libfftw3-dev python3-pip libssl-dev &&\
    apt-get -y install flex libxaw7-dev  &&\
  apt-get -y clean

RUN pip3 install cmake --upgrade
ENV PATH=~/.local/bin:$PATH

RUN git clone https://github.com/opencv/opencv.git /opt/opencv/src && cd /opt/opencv/src && git checkout 4.x &&\
    mkdir /opt/opencv/build && cd /opt/opencv/build  &&\
    cmake -DBUILD_LIST=core ../src &&\
    make -j8 install

# build arrow
RUN apt update &&\
    apt install -y -V ca-certificates lsb-release wget &&\
    wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
    apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb &&\
    apt update &&\
    apt install -y -V libarrow-dev &&\
    apt install -y -V libarrow-glib-dev &&\
    apt install -y -V libarrow-dataset-dev &&\
    apt install -y -V libarrow-dataset-glib-dev &&\
    apt install -y -V libarrow-acero-dev &&\
    apt install -y -V libarrow-flight-dev &&\
    apt install -y -V libarrow-flight-glib-dev &&\
    apt install -y -V libarrow-flight-sql-dev &&\
    apt install -y -V libarrow-flight-sql-glib-dev &&\
    apt install -y -V libgandiva-dev &&\
    apt install -y -V libgandiva-glib-dev &&\
    apt install -y -V libparquet-dev &&\
    apt install -y -V libparquet-glib-dev


ARG sepflags="-O3 -ffast-math -DNDEBUG -DBOOST_DISABLE_ASSERTS -funroll-loops"
# # temp
RUN git config --global http.sslVerify false
# # temp
RUN git clone http://zapad.stanford.edu/SEP-external/sep-iolibs.git /opt/sep-io/src && \
    mkdir /opt/sep-io/build && \
    cd /opt/sep-io/src && git submodule update --init --recursive && \
    cd external/genericIO && git checkout rustam-dev && \
    cd /opt/sep-io/src && git add . &&\
    cd /opt/sep-io/build &&\
    cmake -DCMAKE_INSTALL_PREFIX=/opt/SEP -DCMAKE_CXX_FLAGS="$sepflags" \
    -DBUILD_TEST=ON -DBUILD_SEP=ON -DBUILD_SEGYIO=OFF -DBUILD_GCP=OFF -DBUILD_UTIL=OFF \
    -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_PYTHON=ON ../src &&\
    cd /opt/sep-io/build && \
    make -j8 install

RUN git clone --branch rustam-dev http://cees-gitlab.stanford.edu/ettore88/python-solver.git /opt/python-solver

RUN pip install notebook &&\
    pip install h5py &&\
    pip install numba &&\
    pip install scipy &&\
    pip install -U matplotlib &&\
    apt-get -y update &&\
    apt-get -y install vim

RUN cd /opt && \
    apt-get -y install wget &&\
    wget https://github.com/ispc/ispc/releases/download/v1.17.0/ispc-v1.17.0-linux.tar.gz  &&\
    tar -xvf ispc-v1.17.0-linux.tar.gz &&\
    mv ispc-v1.17.0-linux/bin/ispc /usr/local/bin &&\
    rm -rf ispc-v1.17.0-linux ispc-v1.17.0-linux.tar.gz

RUN git clone https://github.com/pybind/pybind11.git /opt/pybind11/src && \
    mkdir /opt/pybind11/build &&\
    cd /opt/pybind11/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local ../src  &&\
    make -j 4 install

RUN pip install dask distributed dask_jobqueue
RUN pip install "holoviews[recommended]"
RUN pip install numba && pip install pyarrow && pip install segyio

# WORKDIR /home

ENV PYTHONPATH=/opt/SEP/lib/python3.8
ENV PYTHONPATH=/opt/python-solver/GenericSolver/python:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/SEP/lib
ENV PATH=$PATH:/opt/SEP/bin

# EXPOSE 8888

