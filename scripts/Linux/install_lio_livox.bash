#! /bin/bash
if [ `whoami` = "root" ];then
    echo "do not add sudo to run this script"
    exit
fi

echo "Hello ${USER}"

sudo apt install \
build-essential \
libceres-dev \
libsuitesparse-dev \
git \
cmake-qt-gui

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

