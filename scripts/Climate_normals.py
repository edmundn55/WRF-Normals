#!/usr/bin/env python

# Created by Edmund Ng
# Created on September 10,2024

# Load packages
import xarray as xr
from pathlib import Path
import cftime
import pandas as pd
import os, glob, datetime, argparse
# Set global option
xr.set_options(keep_attrs=True)

def get_nth_word_custom_delimiter(string, delimiter, n):
    """
    Function: cut string by delimiter and grab nth element
    
    Input: string
    
    Output: nth element in the string
    """
    # Split string by delimiter
    words = string.split(delimiter)
    # Grab nth element in the string
    if 1 <= n <= len(words):
        return words[n-1]
    else:
        return "Invalid"

def build_parameter(data, output = 'path'):
    """
    Function: Extract parameter to build nested directory from xarray dataset
    
    Input: Xarray dataset
    1) WRFTools*
    or
    2) <parameter>_<start year>
    
    Output: nested directory and scenario as string
    """
    # Build paths for different folder types
    # For WRFTools
    if 'WRFTools' in data.attrs['experiment']:
        par = get_nth_word_custom_delimiter(data.attrs['experiment'], '_', 2) 
        path = 'WRFTools/' + par + '/na24/'
        scen = ''                                                            # Empty for WRFTools

    else:
    # Path of other simulations 
        full_par = data.attrs['experiment']
        force_d = get_nth_word_custom_delimiter(full_par, '_' ,1)              # Forcing dataset
        scen = get_nth_word_custom_delimiter(full_par, '_', 2)                 # Scenario
        grid = get_nth_word_custom_delimiter(full_par, '_', 3)                 # Grid
        if grid == 'NA24':                                                   # convert to small case
            grid = 'na24'
        phys = get_nth_word_custom_delimiter(full_par, '_', 4)                 # Physical configuration
        # Fix issues with directory format -YYYY
        if phys[-5] == '-':
            phys = phys[0:-5]
        else:
            pass
        path = force_d + '/' + grid + '/' + phys + '/'
    # return by conditions
    if output == 'path':
        return path, scen
    elif output == 'par':
        return force_d, phys


def climate_normals(freq, dir_ouput, start, end):
    # Info for function 
    """
    Function:Compute monthly or seasonal normals and create netcdf 
             from monthly average data
                
    Input arguments: 
    dir_input: directory of simulation folder
    dir_out: directory of folder for outputs, currently working directory by default
    start: first year of normal as string
    end: last year of normal as string
    * Both start and end inputs are required for slicing
    freq: Frequency for normals, month = Monthly, season = Seasonal 
    """
   
    # Open datasets with dask
    raw_data = {file.stem :xr.open_dataset(file, chunks={'time':-1}, decode_times=False) for file in Path(os.getcwd()).glob('*monthly.nc')}
    
    # Compute Normals for each dataset
    first_loop = True
    for key in raw_data.keys():
        print('\n', key, freq, 'start')
        data = raw_data[key]
        # Grab first year
        start_date = pd.to_datetime(data.attrs['begin_date'])
        # Grab wrf subcategory
        wrf_cat = get_nth_word_custom_delimiter(data.attrs['description'], ' ', 1)
        # Convert time to datetime64
        data['time'] = pd.date_range(start = start_date, periods = data.sizes['time'], freq = 'MS')
        # slice data if selected time period if necessary
        if start != "" and end != "":
            start_date = pd.to_datetime(start + '-01-01')
            end_date = pd.to_datetime(end + '-12-31')
            data = data.sel(time=slice(start_date, end_date))
        else: 
            pass
        # Add global attribute for normal period
        data.attrs['norm_begin_date'] = str(data.time[0].dt.date.values)
        data.attrs['norm_end_date'] = str(data.time[-1].dt.date.values)
        # Identify start and end year of sliced data   
        if first_loop:
            start_year = str(data.time[0].dt.year.values)
            end_year = str(data.time[-1].dt.year.values)
            path, scen = build_parameter(data, output = 'path')
            # Build output directory 
            if dir_ouput != "":
                out_dir = os.path.join(dir_ouput, path)
            else:
                out_dir = path
            # Check if directory exists and create if false
            if os.path.exists(out_dir) == False:
                os.makedirs(out_dir)
            else:
                print(out_dir, 'Directory exists\n')
            first_loop = False
        # Build full path for output file
        # For missing scenario
        if scen == '':
            out_file = out_dir + '/' + wrf_cat + '_' + freq[0:3] + '-norm-' + start_year + '-' + end_year + '.nc'
        else:
            out_file = out_dir + '/' + wrf_cat + '_' + scen + '_' + freq[0:3] + '-norm-' + start_year + '-' + end_year + '.nc'
        # Check if file exists
        if os.path.isfile(out_file):
            print(out_file, 'File exists\n')

        # Ignore temporary file in folder
        elif 'tmp_' in key:
            print(key, 'skip\n')
        else:
            # Group dataset by months or seasons and compute mean
            data_norm = data.groupby('time.' + freq).mean('time')
            # Export normals as netcdfs
            data_norm.to_netcdf(out_file)
        print('done\n')

# Only excute codes when run as a script
if __name__ == "__main__":
    # Description
    parser = argparse.ArgumentParser(description = 
                                     'Function: Compute monthly or seasonal norms and export as netcdf from monthly data series')
    # Optional arguments
    parser.add_argument('-f', '--freq', type = str, default = 'month', help = 'Frequency: month or season')
    # Directory for output
    parser.add_argument('-d', '--dir', type = str, default = '', help = "Directory of output as string")
    # Period
    parser.add_argument('-p', '--period', type = str, nargs = 2, default = ['', ''], 
                        help = 'Period of interet as string: start year end year')
    # Convert arguments to objects
    args = parser.parse_args()
    # args for periods
    start_y, end_y = args.period
    # clear terminal
    os.system('clear')
    # generate normals
    climate_normals(freq = args.freq, dir_ouput = args.dir, start = start_y, end = end_y)
