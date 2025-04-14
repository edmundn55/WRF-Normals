#!/usr/bin/env python
# coding: utf-8
# Define global setting

import xarray as xr
import numpy as np
from netCDF4 import Dataset
import pandas as pd
# internal export
from regrid import build_wrf_grid, fix_latlon
from wrf import (getvar, ALL_TIMES, to_np,get_cartopy, latlon_coords,Constants)
xr.set_options(keep_attrs=True)

class wrfvars:
    """
    Inputs:
    1. Directory for WRF simulation folder
    2. Directory for subfolder where wrf monthly series are stored
    3. Additional inputs for certain variables
    Shortcuts for getting wrf variables for monthly average series
    1. GHT: Geopotential Height
    2. rss: Surface net downward shortwave flux
    3. rls: Surface net downward longwave flux
    4. LH: Latent Heat
    5. SH: Sensible Heat
    6. PBLH: Planetary boundary layer height
    7. T_PL: Temperature at pressure level
    8. srfc_ps: Surface pressure
    9. srfc_evap: Surface accumulated evaporation
    10. UV: u,v wind components at surface or pressure level
    """
    # get values for time dimension
    def get_time(data):
        """
        Function: Add datetime to the time dimension
        """
        # add values to time dimension
        start_date = pd.to_datetime(data.attrs['begin_date'])
        data['time'] = pd.date_range(start = start_date, periods = data.sizes['time'], freq = 'MS')
        return data
    # list pressure levels and corresponding num_press_levels_stag
    def get_pressure_level(wrf_folder, avg_folder):
        """
        Function: Identify the value at each pressure level (dim: num_press_levels_stag) in wrfplev3d
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfplev3d_d01_monthly.nc", decode_times = False)
        for i in data.num_press_levels_stag.values:
            print(f"num_press_levels_stag {i} : {data['P_PL'].isel(time = 0, num_press_levels_stag = i).values} Pa / "
                  f"{data['P_PL'].isel(time = 0, num_press_levels_stag = i).values/100} hPa")
    # lapse rate
    def get_lapse_rate(wrf_folder, avg_folder, plev1, plev2):
        """
        Function: Compute lapse rate 
        Formula = delta(Temperature)/delta(geopotential height) 
        Units: K/km
        Additional Input(s):
            plev1 : lower pressure level in hPa
            plev2 : higher pressure level in hPa
   
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfplev3d_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # convert hpa to pa for index
        plev1 *= 100
        plev2 *= 100
        # slice data by targeted pressure levels
        data_sliced = data[['GHT_PL', 'T_PL']].where(data['P_PL'].isin([plev1, plev2]), drop = True)
        # get difference of var between pressure levels
        data_sliced = data_sliced.diff(dim = 'num_press_levels_stag')
        # get lapse rate in K/km
        data_sliced = data_sliced.assign(lapse_rate = lambda x: x.T_PL/x.GHT_PL*1000)
        # create attrs for lapse rate
        data_sliced['lapse_rate'] = data_sliced['lapse_rate'].assign_attrs(long_name = f'Lapse Rate between {plev1/100} hPa and {plev2/100} hPa', 
                                                                           units = 'K/km')
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced['lapse_rate'].attrs[attrs] = data.attrs[attrs]
        return data_sliced['lapse_rate']
    # Geopotential Height
    def get_GHT(wrf_folder, avg_folder, plev1):
        """
        Function: Get geopotential height at pressure level
        Units: m
        """
        # convert hpa to pa for index
        plev1 *= 100
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfplev3d_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # extract variables and convert to dgC
        data_sliced = data['GHT_PL'].where(data['P_PL'] == plev1, drop = True)
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Effective Albedo
    def get_albedo(wrf_folder, avg_folder):
        """
        Function: Compute effective albedo
        Formula: Accumulated surface upwelling shortwave radiation/Accumulated surface downwelling shortwave radiation
        Units: unitless
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfrad_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data[['ACSWUPB','ACSWDNB']]
        # get albedo
        data_sliced = data_sliced.assign(albedo = lambda x: x.ACSWUPB/x.ACSWDNB)
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced['albedo'].attrs[attrs] = data.attrs[attrs]
        return data_sliced['albedo']
    # Surface net downward shortwave flux
    def get_rss(wrf_folder, avg_folder):
        """
        Function: Get surface net downward shortwave flux/rss
        Formula: Net Radiation - Net Longwave Radiation
        Units: W/m^2 or J/m^2/s
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfrad_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data[['NetRadiation', 'NetLWRadiation']]
        # get albedo
        data_sliced = data_sliced.assign(rss = lambda x: x.NetRadiation - x.NetLWRadiation)
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced['rss'].attrs[attrs] = data.attrs[attrs]
        return data_sliced['rss']
    # Surface net downward longwave flux
    def get_rls(wrf_folder, avg_folder):
        """
        Function: Get surface net downward longwave flux
        Units: W/m^2 or J/m^2/s
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfrad_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['NetLWRadiation']
        data_sliced = data_sliced.rename('rls')
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Latent Heat
    def get_LH(wrf_folder, avg_folder):
        """
        Function: Get latent heat
        Units: W/m^2 or J/m^2/s
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['LH']
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Sensible heat
    def get_SH(wrf_folder, avg_folder):
        """
        Function: Get sensible heat
        Units: W/m^2 or J/m^2/s
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['HFX']
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Planetary boundary layer height
    def get_PBLH(wrf_folder, avg_folder):
        """
        Function: Get planetary boundary layer height
        Units: m
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['PBLH']
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Surface Pressure
    def get_srfc_ps(wrf_folder, avg_folder, conver_unit = False):
        """
        Function: Get surface pressure
        Units: Pa or hPa 
        Additional input(s):
            conver_unit: True/False (default: False), convert Pa to hPa
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables and convert to hPa
        data_sliced = data[['PSFC']]
        if conver_unit is True:
            data_sliced['PSFC'] = data_sliced['PSFC']/100
            # change units after conversion
            data_sliced['PSFC'] = data_sliced['PSFC'].assign_attrs(units = 'hPa')
        else:
            pass
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Snow 
    def get_snow(wrf_folder, avg_folder):
        """
        Function: Get accumulated snow
        Units: kg/m^-2/s
        """
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrflsm_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['ACSNOW']
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Temperature at pressure levels
    def get_T_PL(wrf_folder, avg_folder, plev1, conver_unit = False):
        """
        Function: Get temperature at pressure level
        Units: K or °C 
        Additional input(s):
            plev1: pressure level in hPa
            conver_unit: True/False (default: False), convert K to °C
        """
        # convert hpa to pa for index
        plev1 *= 100
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfplev3d_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        if conver_unit is True:
            # extract variables and convert to dgC
            data_sliced = data['T_PL'].where(data['P_PL'] == plev1, drop = True) - 273.15
            # change units after conversion
            data_sliced = data_sliced.assign_attrs(units = 'degree Celcius')
        else:
            pass
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # Surface accumulated evaporation
    def get_srfc_evap(wrf_folder, avg_folder, conver_unit = False):
        """
        Function: Get surface accumulated evaporation
        Units: kg/m^2/s or kg/m^2/day
        Additional input(s):
            conver_unit: True/False (default: False), convert kg/m^2/s to kg/m^2/day
        """
        # convert hpa to pa for index
        data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfhydro_d01_monthly.nc", decode_times = False)
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        # extract variables
        data_sliced = data['SFCEVP']
        if conver_unit is True:
            # convert unit to mm/day
            data_sliced = data_sliced * 86400
            # change units after conversion
            data_sliced = data_sliced.assign_attrs(units = 'kg/m^2/day')
        else:
            pass
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        return data_sliced
    # get u,v component
    def get_UV(wrf_folder, avg_folder, plev1 = None):
        """
        Function: Get U,V wind components at surface level (10m) or pressure level
        Units: m/s
        Additional inputs:
            plev1: pressure level in hPa (default: None for surface level)
        """
        # for u,v at pressure level 
        if plev1 != None:
            plev1 *= 100
            data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfplev3d_d01_monthly.nc", decode_times = False)
            data_sliced = data[['U_PL', 'V_PL']].where(data['P_PL'] == plev1, drop = True)
        # for u,v at surface level
        else:
            data = xr.open_dataset(f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc", decode_times = False)
            # extract variables and convert to hPa
            data_sliced = data[['U10', 'V10']]
        # add global attrs back to dataarray
        for attrs in data.attrs:
            data_sliced.attrs[attrs] = data.attrs[attrs]
        # add time values
        data_sliced = wrfvars.get_time(data_sliced)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data_sliced = fix_latlon(data_sliced, wrf_ll)
        return data_sliced
    # prepare dataset for PET cals adapted from Mani Mahdinia
    def get_PET_cal(wrf_folder, avg_folder):
        """
        Function: Get all variables required for PET computation and rename if necessary
        """
        data = xr.open_mfdataset([f"{wrf_folder}geo_em.d01.nc",
                                  f"{wrf_folder}{avg_folder}/wrfsrfc_d01_monthly.nc",
                                  f"{wrf_folder}{avg_folder}/wrfrad_d01_monthly.nc",
                                  f"{wrf_folder}{avg_folder}/wrflsm_d01_monthly.nc",
                                  f"{wrf_folder}{avg_folder}/wrfxtrm_d01_monthly.nc"],
                                 concat_dim = None, compat = 'override', engine = 'netcdf4',
                                 chunks = {'time' : 4, 'lat' : 32*4, 'lon' : 32*4},
                                 join = 'outer', decode_times = False)
        # rename variable for PET calculation
        data = data.rename({'ALBEDO' : 'A', 'NetRadiation' : 'netrad', 'ACGRDFLX' : 'grdflx',
                            'PSFC' : 'ps', 'T2MAX' : 'Tmax', 'T2MIN' : 'Tmin', 'XLAT_M' : 'xlat',
                            'XLONG_M' : 'xlon', 'HGT_M' : 'zs', 'Q2' : 'q2', 'U10' : 'u10', 'V10' : 'v10'})
        # Remove the time dimension from coordinates
        data['xlat'] = data.xlat.isel(Time=0)
        data['xlon'] = data.xlon.isel(Time=0)

        # Change the name of zs units from 'meters MSL' to 'm' in WRF PET dataset
        data['zs'].attrs['units'] = 'm'
        # add time values
        data = wrfvars.get_time(data)
        # add lat lon to wrf
        wrf_ll = build_wrf_grid(f"{wrf_folder}geo_em.d01.nc")
        data = fix_latlon(data, wrf_ll)
        return data