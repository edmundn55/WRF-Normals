#!/bin/bash

# Function: List simulation with missing plev3d

sim_with_plev=$(find $PWD/*/*/*nc | grep -E "plev" | cut -d '/' -f 7 | sort -u)

for item in $sim_with_plev;do
echo $item
printf '\n'
done





