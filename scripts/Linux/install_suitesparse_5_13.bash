#! /bin/bash
if [ `whoami` = "root" ];then
    echo "do not add sudo to run this script"
    exit
fi

echo "Hello ${USER}"
sudo apt-get install -y libgmp-dev libmpfr-dev
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
git checkout 5.13.0
cd SuiteSparse
cd SuiteSparse_config
make MKLROOT=/opt/intel/oneapi/mkl
cd ..
make
sudo make install INSTALL=/usr/local
