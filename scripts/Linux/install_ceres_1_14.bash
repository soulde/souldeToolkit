#! /bin/bash
if [ `whoami` = "root" ];then
    echo "do not add sudo to run this script"
    exit
fi

echo "Hello ${USER}"
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
bash install_mkl.bash
bash install_suitesparse_5_13.bash
git clone https://github.com/ceres-solver/ceres-solver.git
git checkout 1.14.0
cd ceres-solver
mkdir build
cd build 
cmake .. && make && sudo make install


