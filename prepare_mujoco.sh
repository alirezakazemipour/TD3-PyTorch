#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
source ~/mujoco_env/bin/activate
cd ~/pycharm-community-2020.2.2/bin
./pycharm.sh
