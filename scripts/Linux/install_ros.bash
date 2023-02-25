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
python-wstool

echo "source /opt/ros/${ros_ver}/setup.bash" >> ~/.bashrc
source /opt/ros/${ros_ver}/setup.bash
source ~/.bashrc

