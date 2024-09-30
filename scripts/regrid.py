#!/usr/bin/env python
# coding: utf-8

# load packages
import os,glob,sys
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
# wrf phys
phys = ['conus43','new43','new43loc']
# ERA5
era5s = ['ERA5L', 'era5slev']


# Build curvilinear grids for xarray datasets
def build_wrf_grid(geo_file):
    """
    Function: build x,y curvilinear grids from WRF input dataset and geo_em
    Input: path for geo_em file
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
    wrf_proj={'cart_proj':wrf_cart_proj, 'wrf_crs':wrf_xform_crs, 'wrf_lats':wrf_lats, 'wrf_lons':wrf_lons}
    return wrf_proj

# function for grid parameter
def grid_parameter(ds):
    """
    Function: Build parameters for both input and output dataset if necessary
    Inputs: 
    1. ds_in: path of input dataset (the one needs to be regridded) as string
    Outputs:
    1. Parameters: input grid*, input dataset as string
    * nh = Northern Hemisphere
      na = North America
      grid is in km (approximate)
    """
    # for WRF dataset
    if re.search('wrf', ds, re.I):
        grid = 'wrfna24'
        name = 'wrf'
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
        if re.search('*.nc', ds, re.I):
            name = 'era5slev'
        else:
            name = 'era5slev' + '_' + get_nth_word_custom_delimiter(ds.rfind('/')::],'_',2))
    # generate parameters
    return grid, name

# load dataset and build dictionaries for regridding
def build_dataset(dirs):
    """
    Function: Build dictionaries of xarray datasets from directories
    Input: WRF / ERA5 / CERES / Rutgers directories
    Output: xarray datasets in dictionaries
    """
    # For WRF datasets:
    if any('wrf' in wrf_dir for wrf_dir in dirs):
        wrf_raw = {}
        for wrf_d in dirs:
        # open dataset
            wrf = xr.open_dataset(wrf_d,decode_times=False)
            # get forcing dataset and physics configuration
            force_d, phys = build_parameter(wrf,output=None)
            # Grab first year
            start_date = pd.to_datetime(data.attrs['begin_date'])
            # Convert time to datetime64
            wrf['time'] = pd.date_range(start = start_date, periods = wrf.sizes['time'], freq = 'MS')
            # build dictionary
            wrf_raw[force_d+'_'+phys] = wrf
        return wrf_raw
    # For OBS datasets:
    else:
        obs_raw = {}
        for obs_d in dirs:
            # for ERA5L, CERES
            if '*.nc' in obs_d:
                obs = xr.open_mfdataset(obs_d).sortby('time')
                # get obs dataset name
                obs_name = get_nth_word_custom_delimiter(obs_d, '/', 6)
            # for ERA5slev, Rutgers
            else:
                obs = xr.open_dataset(obs_d)
                obs_name_full = get_nth_word_custom_delimiter(obs_d, '/', 7)
                obs_name = get_nth_word_custom_delimiter(obs_name_full, '_', 1)
            # build dictionary
            obs_raw[obs_name] = obs
        return obs_raw

# Fix lat/lon before regridding
def fix_latlon(ds, type_d, wrf_ll = None):
    """
    Function: Fix lat/lon coordinates in datasets before applying regridding
    Inputs: 
    1. ds: xarray dataset/dataarray
    2. type_d: type of dataset in string
    3. wrf_ll: wrf lats and lons in dict (required for fixing wrf dataset)
    Output: xarray dataset/dataarray
    """
    # for WRF dataset
    if type_d == 'wrf':
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
    1. ds_in: input dataset (dataset that undergoes re-gridding) with name and grid added as attrs
    2. ds_out: output dataset (dataset that provides reference grid) with name and grid added as attrs
    3. m: regridding method* as string, patch as default
    * Methods avaiable: https://xesmf.readthedocs.io/en/stable/user_api.html
      Methods definition: https://earthsystemmodeling.org/regrid/#regridding-methods
    **kwargs  keyword arguments
    5. dir_wm: directory of weight matrix if not stored in current working directory
    Outputs:
    1. Regridded xarray dataset/ dataarray
    2. Regridder weight matrix.nc if missing
    """
    # build weight matrix filename
    wm = ds_in.attrs['ds_name'] +'-'+ ds_in.attrs['grid'] + '_' + ds_out.attrs['ds_name'] + '-' + ds_out.attrs['grid'] + '_' + m + '.nc'
    # build full path 
    # if weight matrix is not stored in current working directory
    if 'dir_wm' in kwargs.keys():
        wm = os.path.join(kwargs['dir_wm'],wm)
    else:
        pass
    # check if weight matrix is previously created
    if os.path.isfile(wm):
        regridder = xe.Regridder(ds_in, ds_out, method = m, weights = wm, keep_attrs = True)
    # create weight matrix nc if missing
    else:
        regridder = xe.Regridder(ds_in, ds_out, method = m, keep_attrs = True)
        # build xr dataset for export, adapted from xesmf/frontend.py (Line 759 - 768). DOI: https://doi.org/10.5281/zenodo.4294774
        w = regridder.weights.data
        dim = 'n_s'
        wm_ds = xr.Dataset(
            {'S': (dim, w.data), 'col': (dim, w.coords[1, :] + 1), 'row': (dim, w.coords[0, :] + 1)}
        )
        # add attrs
        wm_ds.attrs['input_grid'] = ds_in.attrs['input_grid']
        wm_ds.attrs['output_grid'] = ds_in.attrs['output_grid']
        wm_ds.to_netcdf(path = wm)
        
    # Perform regridding 
    ds_in_re = regridder(ds_in)
    # Update grid attrs in regridded dataset
    ds_in_re['input_grid'] = ds_in.attrs['input_grid']
    ds_in_re['output_grid'] = ds_in.attrs['output_grid']
    return ds_in_re

# Interpolate to WRF24 grids
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

