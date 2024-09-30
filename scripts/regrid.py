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
    # for CERES
    elif re.search('ceres', ds, re.I):
        grid = 'na111'
        name = 'ceres'
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
        name = 'era5slev'
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

# Interpolate to WRF24 grids
def to_WRF_grid(datasets, wrf_proj, wrf_d = None, m = 'patch'):
    """
    Function: Interpolate input dataset (WRF/ERA5/CERES/Rutgers) to WRF24 grid and perform remapping 
              remap from polar to WRF24 for i.ie. Rutgers Northern Hemisphere 24 km Weekly Snow Cover Extent
              or from WGS1984 to WRF24 for i.e. ERA5 Land, ERA5 plev, ERA5 slev
    Input: WRF / ERA5 / CERES / Rutgers xarray dataarrays in dictionaries,
           WRF projection, coordinate system, lats, lons in dictionary
    Output: xarray dataset(s) in WRF24 grid
    """
    # create new dictionary for remapped dataarrays
    datasets_re = {}
    # for WRF
    if any(phy in name for phy,name in zip(phys,datasets.keys())):
        for name in datasets.keys():
            datasets[name] = datasets[name].assign_coords({'lat':(('south_north','west_east'),wrf_proj['wrf_lats'].values),'lon':(('south_north','west_east'),wrf_proj['wrf_lons'].values)})
            # Generate lat long based on WRF Projection
            xform_pts = wrf_proj['cart_proj'].transform_points(wrf_proj['wrf_crs'],to_np(datasets[name].lon),to_np(datasets[name].lat))
            wrf_x = xform_pts[...,0]
            wrf_y = xform_pts[...,1]
            # insert lat and lon grids into dataset         
            datasets_re[name] = datasets[name].assign_coords({'lat':(('south_north','west_east'),wrf_y),'lon':(('south_north','west_east'),wrf_x)})
        return datasets, datasets_re
    # for obs datasets
    else:
        for name in datasets.keys():
            # for rutgers
            if name == 'rutgers':
                datasets[name] = datasets[name].rename({'latitude':'lat','longitude':'lon'})
                # extract lats/lons
                obs_lat = datasets[name].lat.values
                obs_lon = datasets[name].lon.values.T
                # replace missing values with np.nan and reassign into dataarray
                new_lat = np.where(obs_lat>90,np.nan,obs_lat)
                new_lon = np.where(obs_lon>180,np.nan,obs_lon)
                datasets[name] = datasets[name].assign_coords({'lat':(('x','y'),new_lat),'lon':(('x','y'),new_lon)})
            # rename latitude and longitude to lat/lon for ERA5L and era5slev
            elif any(era5 in name for era5 in era5s):
                datasets[name] = datasets[name].rename({'latitude':'lat','longitude':'lon'})
            # Regridding with WRF dataarray in curvilinear grid instead of WRF grid
            Regridder_obs=xe.Regridder(datasets[name],wrf_d, method = m)
            datasets_re[name]=Regridder_obs(datasets[name],keep_attrs=True)
            # Transform to WRF Projection
            xform_pts = wrf_proj['cart_proj'].transform_points(wrf_proj['wrf_crs'],to_np(datasets_re[name].lon.values),to_np(datasets_re[name].lat.values))
            obs_wrf_x = xform_pts[...,0]
            obs_wrf_y = xform_pts[...,1]
            datasets_re[name]=datasets_re[name].assign_coords({'lat':(('south_north','west_east'),obs_wrf_y),
                                                               'lon':(('south_north','west_east',),obs_wrf_x)})
        return datasets_re

S