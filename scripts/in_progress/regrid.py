#!/usr/bin/env python
# coding: utf-8

# load packages
import os, glob, sys, re
import xesmf as xe
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import cartopy.crs as crs
import matplotlib
from cartopy.feature import NaturalEarthFeature 
import cartopy.feature as cfeature
import datetime
sys.path.append('/project/p/peltier/edmundn/climate_normals_Py/scripts/')
from Climate_normals import get_nth_word_custom_delimiter, build_parameter
from wrf import (getvar, interplevel, vertcross, 
                 CoordPair, ALL_TIMES, to_np,
                 get_cartopy, latlon_coords,
                 cartopy_xlim, cartopy_ylim,
                 Constants,extract_vars)
from matplotlib import pyplot as plt
# Define global setting
xr.set_options(keep_attrs=True)
# geo file for wrfna24 grid
geo_file = '/scratch/p/peltier/mahdinia/wrf/MPIESM12HR_HS_na24_conus43-flk_1979/geo_em.d01.nc'

# Build curvilinear grids for xarray datasets
def build_wrf_grid(geo_file):
    """
    Function: build x,y curvilinear grids from WRF input dataset and geo_em
    Input: path for WRF dataset, path for geo_em file
    Output: dictionary with WRF projection, coordinate system, lats, lons
    """
    # Open geo_em file, get HGT var, cart_proj and lats and lons
    geo = Dataset(geo_file)
    # GMTED2010 30-arc-second topography height
    hgt = getvar(geo, "HGT_M", timeidx = ALL_TIMES)           
    # map projection
    wrf_cart_proj = get_cartopy(hgt)
    # lat, lon of WRF in WRF grid
    wrf_lats, wrf_lons = latlon_coords(hgt)
    # Create WRF Projection
    wrf_globe = crs.Globe(ellipse = None,
                          semimajor_axis = Constants.WRF_EARTH_RADIUS,
                          semiminor_axis = Constants.WRF_EARTH_RADIUS)
    # Define a latitude/longitude coordinate system
    wrf_xform_crs = crs.Geodetic(globe = wrf_globe)
    # Store ouput in dictionary
    wrf_proj={'cart_proj':wrf_cart_proj, 'wrf_crs':wrf_xform_crs}
    wrf_latlon = {'lats' : wrf_lats, 'lons' : wrf_lons}
    return wrf_proj, wrf_latlon

# function for grid parameter
def grid_parameter(ds):
    """
    Function: Build parameters for both input and output dataset if necessary
    Inputs: 
    1. ds_in: path of input dataset (the one needs to be regridded) as string
    Outputs:
    1. Parameters: input grid*, dataset type as string
    * nh = Northern Hemisphere
      na = North America
      grid is in km (approximate)
    """
    # for WRF dataset
    if re.search('wrf', ds, re.I):
        name = 'wrf'
        # find grid type
        if re.search('na24', ds, re.I):
            grid = 'na24'
        elif re.search('wrf24', ds, re.I):
            grid = 'wrf24'
    # for NRCan
    elif re.search('nrcan', ds, re.I):
        grid = 'na12'
        name = 'nrcan'
    # for CERES EBAF-level3b
    elif re.search('ceres', ds, re.I) and re.search('ebaf-level3b', ds, re.I):
        grid = 'na111'
        name = 'ceres-ebaf-level3b'
    # for Rutgers:
    elif re.search('rutger', ds, re.I):
        grid = 'nh24'
        name = 'rutger'
    # for ERA5 Land
    elif re.search('era5l', ds, re.I):
        grid = 'na10'
        name = 'era5l'
    # for ERA5 on pressure levels
    elif re.search('era5plev', ds, re.I):
        grid = 'na28'
        name = 'era5plev'
    # for ERA5 on single levels
    elif re.search('era5slev', ds, re.I):
        grid = 'na28'
        # find subcategory
        if re.search('\*.nc', ds, re.I):
            name = 'era5slev'
        else:
            name = 'era5slev' + '_' + get_nth_word_custom_delimiter(ds[ds.rfind('/')::],'_',2)
    # generate parameters
    return grid, name

# load dataset and build dictionaries for regridding
def build_dataset(dirs):
    """
    Function: load xarray dataset based on file type
    Input: directory
    Output: xarray dataset/ dataarray with ds_type and input_grid added as attrs
    """
        # For WRF datasets:
    if 'wrf' in dirs:
        # open dataset
        ds = xr.open_dataset(dirs, decode_times=False)
        # Grab first year
        start_date = pd.to_datetime(ds.attrs['begin_date'])
        # Convert time to datetime64
        ds['time'] = pd.date_range(start = start_date, periods = ds.sizes['time'], freq = 'MS')
    # For OBS datasets:
    else:
        # for ERA5L, CERES monthly nc
        if '*.nc' in dirs:
            ds = xr.open_mfdataset(dirs).sortby('time')
        # for ERA5slev, Rutgers
        else:
            ds = xr.open_dataset(dirs)
    # add attrs for regridding
    grid, name = grid_parameter(dirs)
    ds.attrs['input_grid'] = grid
    ds.attrs['ds_type'] = name
    ds.attrs['ds_path'] = dirs
    return ds

# Fix lat/lon before regridding
def fix_latlon(ds, wrf_ll = None):
    """
    Function: Fix lat/lon coordinates in datasets before applying regridding
    Inputs: 
    1. ds: xarray dataset/dataarray
    2. type_d: type of dataset in string
    3. wrf_ll: wrf lats and lons in dict (required for fixing wrf dataset)
    Output: xarray dataset/dataarray
    """
    # for WRF dataset
    if 'lat' not in ds.coords and 'latitude' not in ds.coords:
        # check if wrf lats and lons are provided
        if wrf_ll is None:
            print('missing lats lons ')
        else:
            ds_fixed = ds.assign_coords({'lat':(('south_north','west_east'),wrf_ll['lats'].values),'lon':(('south_north','west_east'),wrf_ll['lons'].values)})
    # fixing coordinates name
    elif 'latitude' in ds.coords or 'longitude' in ds.coords:
        #rename coordinates name
        ds_fixed = ds.rename({'latitude':'lat','longitude':'lon'})
    else:
        ds_fixed = ds
    return  ds_fixed

# Function for regridding
def regrid_ds(ds_in, ds_out, m = 'patch', **kwargs):
    """
    Function: perform regridding, reuse regridder weight matrix if found or write to disk as netcdf if missing
    Inputs:
    1. ds_in: input dataset (dataset that undergoes re-gridding) with ds_type and input_grid added as attrs
    2. ds_out: output dataset (dataset that provides reference grid) with ds_type and input_grid added as attrs
    3. m: regridding method* as string, patch as default
    * Methods avaiable: https://xesmf.readthedocs.io/en/stable/user_api.html
      Methods definition: https://earthsystemmodeling.org/regrid/#regridding-methods
    **kwargs  keyword arguments
    4. dir_wm: directory of weight matrix if not stored in current working directory
    5. dir_wm_new: directory of newly create weight matrix, current working directory by default
    Outputs:
    1. Regridded xarray dataset/ dataarray
    2. Regridder weight matrix.nc if missing
    """
    # build weight matrix filename
    wm = ds_in.attrs['ds_type'] +'-'+ ds_in.attrs['input_grid'] + '_' + ds_out.attrs['ds_type'] + '-' + ds_out.attrs['input_grid'] + '_' + m + '.nc'
    # build full path 
    # if weight matrix is not stored in current working directory
    if 'dir_wm' in kwargs.keys():
        wm = os.path.join(kwargs['dir_wm'],wm)
    else:
        pass
    # check if weight matrix is previously created
    if os.path.isfile(wm):
        print('weight matrix exists')
        regridder = xe.Regridder(ds_in, ds_out, method = m, weights = wm)
    # create weight matrix nc if missing
    else:
        print('building weight matrix...')
        regridder = xe.Regridder(ds_in, ds_out, method = m)
        # build xr dataset for export, adapted from xesmf/frontend.py (Line 759 - 768). DOI: https://doi.org/10.5281/zenodo.4294774
        w = regridder.weights.data
        dim = 'n_s'
        wm_ds = xr.Dataset(
            {'S': (dim, w.data), 'col': (dim, w.coords[1, :] + 1), 'row': (dim, w.coords[0, :] + 1)}
        )
        # add attrs
        wm_ds.attrs['input_grid'] = ds_in.attrs['input_grid']
        wm_ds.attrs['output_grid'] = ds_out.attrs['input_grid']
        # writing weight matrix to disk
        # for storing weight matrix to other location
        if 'dir_wm_new' in kwargs.keys():
            # check if folder exists
            if os.path.exists(kwargs['dir_wm_new']) is False:
                os.makedirs(kwargs['dir_wm_new'])
            else:
                pass
            wm = os.path.join(kwargs['dir_wm_new'], wm)
        wm_ds.to_netcdf(path = wm)
        
    # Perform regridding 
    print('regridding...')
    ds_in_re = regridder(ds_in, keep_attrs = True)
    # Update grid attrs in regridded dataset
    ds_in_re['input_grid'] = ds_in.attrs['input_grid']
    ds_in_re['output_grid'] = ds_out.attrs['input_grid']
    return ds_in_re

# Interpolate to WRF24 grids
def to_WRF_grid(ds, wrf_proj):
    """
    Function: Interpolate input dataset (WRF/ERA5/CERES/Rutgers) to WRF24 grid and perform remapping 
              remap from polar to WRF24 for i.ie. Rutgers Northern Hemisphere 24 km Weekly Snow Cover Extent
              or from WGS1984 to WRF24 for i.e. ERA5 Land, ERA5 plev, ERA5 slev
    Input: WRF / ERA5 / CERES / Rutgers xarray dataset/dataarray,
           WRF projection, coordinate system 
    Output: xarray dataset/dataarray in WRF24 grid
    """
    # Transform to WRF Projection
    xform_pts = wrf_proj['cart_proj'].transform_points(wrf_proj['wrf_crs'],to_np(ds.lon.values),to_np(ds.lat.values))
    wrf_x = xform_pts[...,0]
    wrf_y = xform_pts[...,1]
    ds_wrf=ds.assign_coords({'lat':(('south_north','west_east'),wrf_y),
                                                       'lon':(('south_north','west_east',),wrf_x)})
    return ds_wrf

def find_tem_freq(ds_in):
    """
    Functions: identify temporal frequency
    Input: 
    1. ds_in: xarray dataset/dataarray
    Output:
    !. temporal frequency as string
    """
    freq = xr.infer_freq(ds_in.time)
    # if xr.infer_freq does not work
    if freq is None:
        delta = pd.to_timedelta(np.diff(ds_in.time)).mean()
        if delta.days <=31 and delta.days>=28:
            freq = 'MS'
        elif delta.days == 7:
            freq = 'W'
    else:
        pass
    # return freq name based on results
    if 'MS' in freq or 'ME' in freq:
        ds_freq = 'monthly'
    elif 'W' in freq:
        ds_freq = 'weekly'
    elif 'Q' in freq:
        ds_freq = 'seasonal'
    elif 'Y' in freq:
        ds_freq = 'yearly'
    return ds_freq

# Function to build filename of exported netcdf
def build_regridded_nc_filename(ds_in):
    """
    Function: build filename for exported regridded nc
    Inputs:
    1. ds_in: xarray dataset/dataarray of output dataset with ds_name, input_grid, output_grid, ds_path as attrs
    Output:
    Filename
    """
    # for WRF
    if re.search('wrf', ds_in.attrs['ds_path'], re.I):
        # for WRF norm
        if re.search('norm', ds_in.attrs['ds_path']):
            filename = ds_in.attrs['ds_path'].replace(ds_in.attrs['input_grid'],ds_in.attrs['output_grid'])
        # for wrf monthly series
        else:
            # build path 
            path, scen = build_parameter(ds_in)
            # replace grid in path
            path_new = path.replace(ds_in.attrs['input_grid'],ds_in.attrs['output_grid'])
            # find period
            start_year = str(ds_in.time[0].dt.year.values)
            end_year = str(ds_in.time[-1].dt.year.values)
            # get wrf subcategory
            wrf_cat = get_nth_word_custom_delimiter(ds_in.attrs['description'], ' ', 1)
            # name for netcdf
            filename_sub = wrf_cat + '_' + scen + '_' + 'monthly-' + start_year + '-' + end_year +'.nc' 
            # build filename
            filename = os.path.join(path_new, filename_sub)
    # for obs dataset 
    elif re.search('Data', ds_in.attrs['ds_path']):
        # for dataset built from multiple files
        if re.search('\*.nc', ds_in.attrs['ds_path']):
            # find period
            start_year = str(ds_in.time[0].dt.year.values)
            end_year = str(ds_in.time[-1].dt.year.values)
            # frequency
            freq = find_tem_freq(ds_in)
            # build filename
            filename = ds_in.attrs['ds_name'] + '_' + ds_in.attrs['output_grid'] + '_' + freq + '-' + start_year +'-' + end_year +'.nc'
        # for single file
        else:
            filename = ds_in.attrs['ds_path'].replace(ds_in.attrs['input_grid'],ds_in.attrs['output_grid'])
    return filename

# Function for automated regridding
def main(dir_in, dir_out, **kwargs):
    """
    Function: automated regridding based on file paths and assign output directories if necessary
    Inputs:
    1. dir_in: input dataset (dataset that undergoes re-gridding)
    2. dir_out: output dataset (dataset that provides reference grid)
    **kwargs, keyword arguements
    3. 
    """