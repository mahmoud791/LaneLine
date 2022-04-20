#!/bin/bash

# Ask the user for login details
read -p 'Specfiy the input video path: ' inputPathVar
read -p 'Specfiy the output video path: ' outputPathVar
read -p 'Use debugging mode [y/n]: ' debugVar

echo -e "installing the dependants packges !! \n"

pip install opencv-python
pip install moviepy
pip install matplotlib
pip install sklearn


echo -e "\n Starting execution , it may take time , please be patient \n "
python lane.py $inputPathVar $outputPathVar $debugVar > /dev/null
echo -e "\n execution finished \n "
