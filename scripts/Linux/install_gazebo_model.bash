#! /bin/bash
cd ~/.gazebo
mkdir models
cd models
wget http://file.ncnynl.com/ros/gazebo_models.txt
wget -i gazebo_models.txt
ls model.tar.g* | xargs -n1 tar xzvf

rm gazebo_models.txt
rm model.tar.g*

