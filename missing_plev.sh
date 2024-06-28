#!/bin/bash

# Function: Find simulation with missing plev3d and build slurm command

# Simulation without plev

sim_without_plev=$(find $PWD/*/monthly*  -type d '!' -exec sh -c 'ls -1 "{}"|egrep -i -q "*plev*"' ';' -print | cut -d '/' -f 7-8 | sort -u)

for item in ${sim_without_plev};do
sim_name=$(echo ${item} | cut -d '/' -f 1)
start_year=$(echo ${item}  | cut -d '/' -f 2 | cut -d '_' -f 3)
end_year=$( echo ${item} | cut -d '/' -f 2 | cut -d '_' -f 4) 
echo ${sim_name},${start_year},${end_year}
printf '\n'

done






