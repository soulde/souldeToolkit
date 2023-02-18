#!/bin/bash

if [ `whoami` = "root" ];then
    echo "do not add sudo to run this script"
    exit
fi

echo "Hello ${USER}"
sudo sh -c 'echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt upgrade -y
ros_ver=""
case $(lsb_release -sc) in
    xenial ) 
    ros_ver="kinetic"
    ;;
    bionic )
    ros_ver="melodic"
    ;;
    *)
    echo "version not support"
    exit
esac

sudo apt-get install -y \
ros-${ros_ver}-desktop-full \
python-rosinstall \
python-rosinstall-generator \
python-wstool \
build-essential \
libceres-dev \
libsuitesparse-dev \
git \
cmake-qt-gui
echo "source /opt/ros/${ros_ver}/setup.bash" >> ~/.bashrc
source /opt/ros/${ros_ver}/setup.bash
source ~/.bashrc
sudo apt autoremove -y

echo "**************************"
echo "Install Livox-SDK"
echo "**************************"

git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK
cd build && cmake ..
make -j2
sudo make install
cd ../..
rm -r -f Livox-SDK

echo "**************************"
echo "Install livox_ros_driver"
echo "**************************"
cd
mkdir -p livox_ros_driver_pkg/src
cd livox_ros_driver_pkg/src
catkin_init_workspace
cd ..
catkin_make
cd src
git clone https://github.com/Livox-SDK/livox_ros_driver.git
cd ..
catkin_make
cd ..
echo "source /home/${USER}/livox_ros_driver_pkg/devel/setup.bash" >> ~/.bashrc
source /home/${USER}/livox_ros_driver_pkg/devel/setup.bash
source ~/.bashrc
echo "**************************"
echo "Install LIO-Livox"
echo "**************************"
cd 
ws_dir="rm_ws"
mkdir -p ${ws_dir}/src
cd ${ws_dir}/src
catkin_init_workspace
cd ..
catkin_make
cd src
git clone https://github.com/Livox-SDK/LIO-Livox.git
cd ..
catkin_make
echo "**************************"
echo "ALL DONE"
echo "**************************"

