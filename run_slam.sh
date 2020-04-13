#!/bin/bash

script_dir=$(dirname $0)
cur_dir=$(pwd)
cmd="/home/eatelephant/anaconda3/envs/slam/bin/python slam.py"
data_folder="Data/Color/03/image_2"
calib_file="Data/Color/03/calib.txt"

cd ${script_dir}

run_cmd="${cmd} ${data_folder} ${calib_file}"
${run_cmd}

cd ${cur_dir}

