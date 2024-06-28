#!/bin/bash

# Function: Find simulation with missing plev3d and build slurm command for arconfig

# Simulation without plev

sim_without_plev=$(find $PWD/*/monthly*  -type d '!' -exec sh -c 'ls -1 "{}"|egrep -i -q "*plev*"' ';' -print | cut -d '/' -f 7-8 | sort -u)

# build parameters for slurm  

for item in ${sim_without_plev};do

# Name of simulation 
sim_name=$(echo ${item} | cut -d '/' -f 1)

# Starting year
start_year=$(echo ${item}  | cut -d '/' -f 2 | cut -d '_' -f 3)

# Ending year
end_year=$( echo ${item} | cut -d '/' -f 2 | cut -d '_' -f 4) 

# Sequence for TAGS
all_year=$(seq ${start_year} ${end_year})

# For WRF* simulations

if [[ ${sim_name} == "WRF"* ]]; then

#  SRC (Directory in scratch)

  src_name=$(find /scratch/p/peltier/mahdinia/ERAI_AND_ERA5_RUN_CASES/* \
  /scratch/a/aerler/aerler/Mani/ERAI_AND_ERA5_RUN_CASES/* -type d -name ${sim_name})

# DST (Directory at archive)

  dst_name=/archive/p/peltier/mahdinia/ERAI_AND_ERA5_RUN_CASES/${sim_name}

# For the other simulations

else

# SRC
  src_name=$(find /scratch/p/peltier/mahdinia/wrf/* \
  /scratch/a/aerler/aerler/Mani/wrf/* -type d -name ${sim_name}*${start_year})

# Update sim_name

  sim_name=$(echo ${src_name} | rev | cut -d '/' -f 1 | rev)

# DST
  
dst_name=/archive/p/peltier/mahdinia/wrf/${sim_name}
fi

# slurm command

slurm_command="--export=TAGS=\"${all_year}\",MODE=RETRIEVE,INTERVAL=YEARLY,SCR=\"${src_name}\",\
DST=\"${dst_name}\",DATASET=DIAGS"

echo ${sim_name}
echo ${slurm_command}
echo -e '\n'
done






