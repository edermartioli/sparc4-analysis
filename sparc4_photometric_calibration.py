"""
    Created on Feb 28 2025
    
    Description: A tool to obtain a photometric calibraiton for SPARC4
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:

    python sparc4_photometric_calibration.py --input="/Users/eder/Science/sparc4-analysis/data/photcalib_phot/*.csv"
    python sparc4_photometric_calibration.py --input="/Users/eder/Science/sparc4-analysis/data/photcalib_polar/*.csv"

    python sparc4_photometric_calibration.py --input="/Users/eder/Science/sparc4-analysis/data/photcalib_phot/hats-9_20240615_master-catalog_clean.csv" --output="/Users/eder/Science/sparc4-analysis/data/photcalib_phot/hats-9_20240615_master-catalog_clean.json" --output_catalog=/Users/eder/Science/sparc4-analysis/data/photcalib_phot/hats-9_20240615_master-catalog_clean_new.csv
        
        
    python sparc4_photometric_calibration.py --input="/Users/eder/Science/sparc4-analysis/data/photcalib_phot/hats-9_20240615_master-catalog_clean.csv" --input_solution=/Users/eder/Science/sparc4-analysis/data/photcalib_phot/hats-9_20240615_master-catalog_clean.json
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os
from optparse import OptionParser

import glob

import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from astropy.table import Table, join, Column
from scipy import optimize
from uncertainties import ufloat

from astropy import units as u
from astroplan import Observer
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

import json
import emcee
import corner


def color(dmag, airmass, coeffs, time_from_midnight=None) :

    """ Function to calculate magnitude difference "color" model
    
    Parameters
    ----------
    dmag: np.array()
        magnitude difference
    airmass: np.array()
        air mass
    coeffs: np.array()
        color model coefficients
    time_from_midnight: np.array(), optional
        time from local solar midnight
    Returns
        dmag_model: np.array()
            magnitude difference model
    -------
    """
    
    if time_from_midnight is not None :
        dmag_model = coeffs[0]  + coeffs[1] * dmag + coeffs[2] * airmass + coeffs[3] * time_from_midnight
    else :
        #dmag_model = coeffs[0]  + coeffs[1] * dmag + coeffs[2]*airmass + coeffs[3] * airmass * dmag
        dmag_model = coeffs[0]  + coeffs[1] * dmag + coeffs[2] * airmass
        
    return dmag_model

def magnitude_model(mag, dmag, airmass, coeffs, time_from_midnight=None) :

    """ Function to calculate magnitude model
    
    Parameters
    ----------
    mag: np.array()
        magnitude
    dmag: np.array()
        magnitude difference
    airmass: np.array()
        air mass
    coeffs: np.array()
        mag model coefficients
    time_from_midnight: np.array(), optional
        time from local solar midnight
    Returns
        mag_model: np.array()
            magnitude model
    -------
    """
    
    if time_from_midnight is not None :
        mag_model = coeffs[0] + coeffs[1] * dmag + coeffs[2]*airmass + coeffs[3]  * time_from_midnight + mag
    else :
        #mag_model = coeffs[0] + coeffs[1] * dmag + coeffs[2]*airmass + coeffs[3] * airmass * dmag + mag
        mag_model = coeffs[0] + coeffs[1] * dmag + coeffs[2] * airmass + mag
        
    return mag_model


def color_err(mag1, emag1, mag2, emag2) :
    """ Function to propagate errors in magnitude difference (m1-m2)
    
    Parameters
    ----------
    mag1: np.array()
        1st magnitude
    emag1: np.array()
        1st magnitude errors
    mag2: np.array()
        2nd magnitude
    emag2: np.array()
        2nd magnitude errors
    Returns
        edmag: np.array()
            color magnitude errors
    -------
    """
    dmag = mag1 - mag2
    edmag = np.full_like(dmag,np.nan)
    for i in range(len(edmag)) :
        umag1 = ufloat(mag1[i],emag1[i])
        umag2 = ufloat(mag2[i],emag2[i])
        diff = umag1 - umag2
        edmag[i] = diff.std_dev
    return edmag

def errfunc(coeffs, obs_mags, obs_emags, mags, airmass, ncoeffs=3, nbands=4, time_from_midnight=None, return_errors=False) :

    """ Function calculate the wieghted residuals
    
    Parameters
    ----------
    coeffs: np.array()
        full model coefficients
    obs_mags: np.array()
        observed magnitudes
    obs_emags: np.array()
        observed magnitude errors
    mags: np.array()
        nominal magnitudes from reference catalog
    airmass: np.array()
        air mass
    ncoeffs: int
        total number of coefficients in the model
    nbands: int
        number of photometric bands
    time_from_midnight: np.array(), optional
        time from local solar midnight
    return_errors: np.array(), optional
        to return errror, useful for MCMC
    Returns
        residuals, errors: np.array(), np.array()
            residuals weighted by the errors and errors
        
    -------
    """

    residuals = np.array([])
    errors = np.array([])
    ncolors = nbands-1
    
    for i in range(ncolors) :
        
        dmag = mags[i] - mags[i+1]
        obs_dmag = obs_mags[i] - obs_mags[i+1]
        
        i1 = ncoeffs*i
        i2 = ncoeffs*(i+1)
        loc_coeffs = coeffs[i1:i2]
        err = color_err(mags[i],obs_emags[i],mags[i+1],obs_emags[i+1])
        res = (obs_dmag - color(dmag, airmass, loc_coeffs, time_from_midnight=time_from_midnight)) / err
        
        residuals = np.append(residuals,res)
        if return_errors :
            errors = np.append(errors,err)

    # Fit
    for i in range(nbands) :

        if i == nbands-1 :
            dmag =  mags[i-1] - mags[i]
        else :
            dmag = mags[i] - mags[i+1]
        
        i1 = ncolors * ncoeffs + ncoeffs*i
        i2 = ncolors * ncoeffs + ncoeffs*(i+1)
        loc_coeffs = coeffs[i1:i2]
        res = (obs_mags[i] - magnitude_model(mags[i], dmag, airmass, loc_coeffs, time_from_midnight=time_from_midnight)) / obs_emags[i]
        residuals = np.append(residuals,res)
        if return_errors :
            errors = np.append(errors,obs_emags[i])

    if return_errors :
        return residuals, errors
    else :
        return residuals


def calculate_time_from_midnight(ra, dec, time_bjd, equinox=2000., longitude=-45.5825, latitude=-22.53444, altitude=1864) :
    
    """ Function calculate time from local solar midnight
    
    Parameters
    ----------
    ra: np.array()
        right ascensions
    dec: np.array()
        declinations
    time_bjd: np.array()
        barycentric julian dates
    time_bjd: np.array()
        nominal magnitudes from reference catalog
    equinox: float
        equinox
    longitude: float
        gegraphic longitude [deg]
    latitude: float
        geographic latitude [deg]
    altitude: float
        observatory elevation in [m]
    Returns
        time_from_midnight: np.array()
            time difference between the observed time and local solar midnight
    -------
    """
    # se equinox
    str_equinox = "J{:.1f}".format(equinox)
    
    # set observatory location
    observatory_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude*u.m)
    
    # set observer object
    observer = Observer(location=observatory_location, timezone='UTC')
    
    # set obstime
    obstime = Time(time_bjd, format='jd', scale='utc', location=observatory_location)
    
    # set target coords
    target = SkyCoord(ra, dec, unit=(u.deg, u.deg),frame='icrs', equinox=str_equinox)
    
    # calculate time at solar midnight
    meianoite = observer.midnight(obstime)
    
    # calculate time difference between the observed time and midnight
    dt =  (obstime - meianoite)*24
    
    # cast results in a numpy array
    time_from_midnight = np.array(dt.value)
    
    return time_from_midnight



def initialize_phot_calibration (filename, solution="", bands=['g','r','i','z'], use_time_from_midnight=True) :

    """ Function to initialize photometric calibration
    
    Parameters
    ----------
    filename: str
        input file name (*.csv) containing the photometric data
    solution: str
        solution json file name
    bands: list [str,str,str,str]
        photometric bands
    use_time_from_midnight: bool
        whether to include time from midnight in the fit

    Returns
        loc: dict
            dictionary container for the photometric calibration data
    -------
    """
    
    # initialize dict container
    loc = {}
    
    # load input data from csv file
    tbl = ascii.read(filename)

    # print table for checking
    print(tbl)

    # initialize magnitude variables
    obs_mags, obs_emags = [], []
    mags, emags = [], []
    
    keep  = np.isfinite(np.array(tbl['obs_mag_g'].data))
    # make sure to avoid NaNs
    for band in bands :
        keep &= np.isfinite(np.array(tbl['obs_mag_{}'.format(band)].data))
        keep &= np.isfinite(np.array(tbl['obs_emag_{}'.format(band)].data))
        keep &= np.isfinite(np.array(tbl['mag_{}'.format(band)].data))
        keep &= np.isfinite(np.array(tbl['emag_{}'.format(band)].data))

    # initialize time from midnight variable
    time_from_midnight = None
    # load time data
    time_bjd = tbl['time_bjd'].data[keep]
    # load airmass data
    airmass = tbl['airmass'].data[keep]
    
    # load magnitude data from input table
    for band in bands :
        obs_mags.append(np.array(tbl['obs_mag_{}'.format(band)].data)[keep])
        obs_emags.append(np.array(tbl['obs_emag_{}'.format(band)].data)[keep])
        mags.append(np.array(tbl['mag_{}'.format(band)].data)[keep])
        emags.append(np.array(tbl['emag_{}'.format(band)].data)[keep])

    # update coefficients for more complex models
    if use_time_from_midnight :
        # calculate time from midnight for all observations
        time_from_midnight = calculate_time_from_midnight(tbl['ra'].data[keep], tbl['dec'].data[keep], tbl['time_bjd'].data[keep])
    
    # get number of bands
    nbands = len(bands)
    # define number of colors as the number of bands minus one.
    ncolors = nbands-1
    # initialize coefficients for the simplest model
    ncoeffs = 3
    guess_coeffs = [0.0001, 1.0000001, 0.0000001,  # g-r
                    0.0001, 1.0000001, 0.0000001,  # r-i
                    0.0001, 1.0000001, 0.0000001,  # i-z
                    0.0001, 1.0000001, 0.0000001,  # g
                    0.0001, 1.0000001, 0.0000001,  # r
                    0.0001, 1.0000001, 0.0000001,  # i
                    0.0001, 1.0000001, 0.0000001]  # z

    # update coefficients for more complex models
    if use_time_from_midnight :
        # reset number of coefficients
        ncoeffs = 4
        # guess coefficients
        guess_coeffs = [0.0001, 1.0000001, 0.0000001, 0.0000001,  # g-r
                        0.0001, 1.0000001, 0.0000001, 0.0000001,  # r-i
                        0.0001, 1.0000001, 0.0000001, 0.0000001,  # i-z
                        0.0001, 1.0000001, 0.0000001, 0.0000001,  # g
                        0.0001, 1.0000001, 0.0000001, 0.0000001,  # r
                        0.0001, 1.0000001, 0.0000001, 0.0000001,  # i
                        0.0001, 1.0000001, 0.0000001, 0.0000001]  # z
    
    # when a solution is given, set as initial guess
    if solution != "" :
        with open(solution, 'r') as json_file:
            photcalib = json.load(json_file)
        photcalib["coeffs"] =  np.array(photcalib["coeffs"])
    
        bands = photcalib['bands']
        ncoeffs = photcalib['ncoeffs']
        nbands = photcalib['nbands']
        ncolors = photcalib['ncolors']
        guess_coeffs = photcalib['coeffs']
        use_time_from_midnight = photcalib['use_time_from_midnight']
        
    loc["use_time_from_midnight"] = use_time_from_midnight

    loc["bands"] = bands
    loc["nbands"] = nbands
    loc["ncolors"] = ncolors

    loc["time_bjd"] = time_bjd
    loc["airmass"] = airmass
    loc["time_from_midnight"] = time_from_midnight

    loc["obs_mags"] = obs_mags
    loc["obs_emags"] = obs_emags
    loc["mags"] = mags
    loc["emags"] = emags

    loc["ncoeffs"] = ncoeffs
    loc["coeffs"] = guess_coeffs
    
    # make sure to include data from all fields
    for col in tbl.colnames:
        if col not in loc.keys() :
            loc[col] = tbl[col].data[keep]
            
    return loc


def run_photometric_calibration(photcalib, max_airmass=3.0, n_sigma_clip=4.0, niter=5, verbose=True) :
    
    """ Function to initialize photometric calibration
    
    Parameters
    ----------
    photcalib: dict
        dictionary container for the photometric calibration data
    max_airmass: float
        maximum value of airmass
    n_sigma_clip: int
        number of sigmas to clip data
    niter: int
        number of iterations in the sigma clip <-> fit process
    verbose: bool
        print information
    Returns
        photcalib: dict
            dictionary container for the photometric calibration data
    -------
    """
    
    for i in range(niter) :
    
        # run the OLS fit
        pfit, pcov, infodict, errmsg, success = optimize.leastsq(errfunc, photcalib["coeffs"], args=(photcalib["obs_mags"],
                                                                             photcalib["obs_emags"],
                                                                             photcalib["mags"],
                                                                             photcalib["airmass"],
                                                                             photcalib["ncoeffs"],
                                                                             photcalib["nbands"],
                                                                             photcalib["time_from_midnight"]),full_output=True)

        # print fit parameters for checking
        if verbose :
            print(pfit)

        # get residuals
        residuals = errfunc(pfit, photcalib["obs_mags"],
                                  photcalib["obs_emags"],
                                  photcalib["mags"],
                                  photcalib["airmass"],
                                  ncoeffs=photcalib["ncoeffs"],
                                  nbands=photcalib["nbands"],
                                  time_from_midnight=photcalib["time_from_midnight"])
        
        
        # apply first filter
        keep = photcalib["airmass"] < max_airmass
        
        for j in range(photcalib["nbands"]) :
            if j == photcalib["nbands"]-1 :
                dmag =  photcalib["mags"][j-1] - photcalib["mags"][j]
            else :
                dmag = photcalib["mags"][j] - photcalib["mags"][j+1]
                
            i1 = photcalib["ncolors"] * photcalib["ncoeffs"] + photcalib["ncoeffs"]*j
            i2 = photcalib["ncolors"] * photcalib["ncoeffs"] + photcalib["ncoeffs"]*(j+1)
            
            loc_coeffs = pfit[i1:i2]
            
            magmodel = magnitude_model(photcalib["mags"][j], dmag, photcalib["airmass"], loc_coeffs, time_from_midnight=photcalib["time_from_midnight"])
            
            residuals = photcalib["obs_mags"][j] - magmodel
        
            sig = np.std(residuals)
        
            keep &= np.abs(residuals) < n_sigma_clip * sig
        
            if verbose :
                print("ITERATION # {} of {} Band: {} -> RMS = {}".format(i+1,niter,photcalib["bands"][j], sig))
        
        # Print information
        if verbose :
            npoints = len(photcalib["airmass"])
            npointsleft = len(photcalib["airmass"][keep])
            npoints_rejected = npoints - npointsleft
            print("Applying a {}-sigma clip -> Number of points rejected: {}. Total number of points left: {}".format(n_sigma_clip,npoints_rejected,npointsleft))
     
        # Update data, applying the sigma clip
        photcalib['gaiadr2_id'] = photcalib['gaiadr2_id'][keep]
        photcalib['ra'] = photcalib['ra'][keep]
        photcalib['dec'] = photcalib['dec'][keep]
        photcalib['pmRA'] = photcalib['pmRA'][keep]
        photcalib['pmDE'] = photcalib['pmDE'][keep]
        photcalib['epoch_yr'] = photcalib['epoch_yr'][keep]
        photcalib["time_bjd"] = photcalib["time_bjd"][keep]
        photcalib["airmass"] = photcalib["airmass"][keep]
        if photcalib["time_from_midnight"] is not None :
            photcalib["time_from_midnight"] = photcalib["time_from_midnight"][keep]
        for j in range(photcalib["nbands"]) :
            photcalib["obs_mags"][j] = photcalib["obs_mags"][j][keep]
            photcalib["obs_emags"][j] = photcalib["obs_emags"][j][keep]
            photcalib["mags"][j] = photcalib["mags"][j][keep]
            photcalib["emags"][j] = photcalib["emags"][j][keep]
        photcalib["coeffs"] = pfit

        errors = []
        try :
            pcov *= (residuals**2).sum()/(len(residuals)-len(pfit))
            for i in range(len(pfit)):
                errors.append(np.absolute(pcov[i][i])**0.5)
            photcalib["coeffs_err"] = errors
        except :
            for i in range(len(pfit)):
                errors.append(0.)
        photcalib["coeffs_err"] = np.array(errors)
        print("errors:",errors)
        
    return photcalib


def plot_phot_calib_results(photcalib, color_labels=['darkblue','darkgreen','darkorange','darkred']) :

    # loop over each band to do plots
    for i in range(photcalib["nbands"]) :

        # set index range of coefficients
        i1 = photcalib["ncolors"] * photcalib["ncoeffs"] + photcalib["ncoeffs"]*i
        i2 = photcalib["ncolors"] * photcalib["ncoeffs"] + photcalib["ncoeffs"]*(i+1)
    
        color_label = ""
        # get delta mags
        if i == photcalib["nbands"]-1 :
            dmag =  photcalib["mags"][i-1] - photcalib["mags"][i]
            color_label = "{}-{}".format(photcalib["bands"][i-1],photcalib["bands"][i])
        else :
            dmag = photcalib["mags"][i] - photcalib["mags"][i+1]
            color_label = "{}-{}".format(photcalib["bands"][i],photcalib["bands"][i+1])

        # set coefficients
        coeffs = photcalib["coeffs"][i1:i2]
    
        # calculate magnitude model
        magmodel = magnitude_model(photcalib["mags"][i], dmag, photcalib["airmass"], coeffs, time_from_midnight=photcalib["time_from_midnight"])

        # calculate the correction term
        #corr_term = coeffs[0] + coeffs[1] * dmag + coeffs[3] * photcalib["time_from_midnight"] + photcalib["mags"][i]
        #corr_term = coeffs[0] + coeffs[1] * dmag + coeffs[2] * photcalib["airmass"] + photcalib["mags"][i]
        #mag_m = magmodel - corr_term
        
        #timesort = np.argsort(photcalib["time_bjd"])
        #plt.plot(photcalib["time_bjd"][timesort], (coeffs[2] * photcalib["airmass"])[timesort],"-",color="k")
        #plt.plot(photcalib["time_bjd"][timesort], (photcalib["obs_mags"][i]-corr_term)[timesort],".",color=color_labels[i],alpha=0.7)
        #plt.xlabel(r"time [BJD]",fontsize=20)
        
        #amsort = np.argsort(photcalib["airmass"])
        #plt.plot(photcalib["airmass"][amsort], mag_m[amsort], "-",color="k")
        #plt.plot(photcalib["airmass"][amsort], (photcalib["obs_mags"][i]-corr_term)[amsort],".",color=color_labels[i],alpha=0.7)
        #plt.xlabel(r"airmass",fontsize=20)
        
        #tfmsort = np.argsort(photcalib["time_from_midnight"])
        #plt.plot(photcalib["time_from_midnight"][tfmsort], mag_m[tfmsort], "-",color="k")
        #plt.plot(photcalib["time_from_midnight"][tfmsort], (photcalib["obs_mags"][i]-corr_term)[tfmsort],".",color=color_labels[i],alpha=0.7)
        #plt.xlabel(r"Time from midnight (h)",fontsize=20)
        
        colorsort = np.argsort(dmag)
        plt.errorbar(dmag[colorsort],(photcalib["obs_mags"][i]-magmodel)[colorsort],yerr=photcalib["obs_emags"][i][colorsort],fmt=".",alpha=0.7)
        plt.xlabel(r"{} (mag)".format(color_label),fontsize=20)
        plt.ylabel(r"{} - {}$_m$ (mag)".format(photcalib["bands"][i],photcalib["bands"][i]),fontsize=20)

        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.minorticks_on()
        plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        plt.show()


def save_solution(photcalib, output_json="", output_csv="") :

    loc = {} # set output dict
    time_bjd, airmass = photcalib['time_bjd'], photcalib['airmass']
    
    loc["min_time_bjd"] = np.min(time_bjd)
    loc["max_time_bjd"] = np.max(time_bjd)
    loc["mean_time_bjd"] = np.nanmean(time_bjd)
    
    loc["min_airmass"] = np.min(airmass)
    loc["max_airmass"] = np.max(airmass)
    loc["mean_airmass"] = np.nanmean(airmass)
    loc["std_airmass"] = np.nanstd(airmass)

    if photcalib['time_from_midnight'] is not None :
        time_from_midnight = photcalib['time_from_midnight']
        loc["min_time_from_midnight"] = np.min(time_from_midnight)
        loc["max_time_from_midnight"] = np.max(time_from_midnight)
        loc["mean_time_from_midnight"] = np.nanmean(time_from_midnight)
        loc["std_time_from_midnight"] = np.nanstd(time_from_midnight)

    for band in photcalib['bands'] :
        mag = photcalib['mag_{}'.format(band)][i]
        loc["min_{}mag".format(band)] = np.nanmin(mag)
        loc["max_{}mag".format(band)] = np.nanmax(mag)
        loc["mean_{}mag".format(band)] = np.nanmean(mag)
        loc["std_{}mag".format(band)] = np.nanstd(mag)

    loc["use_time_from_midnight"] = photcalib["use_time_from_midnight"]
    loc["bands"] = photcalib["bands"]
    loc["nbands"] = photcalib["nbands"]
    loc["ncolors"] = photcalib["ncolors"]

    loc["ncoeffs"] = photcalib["ncoeffs"]
    loc["coeffs"] = list(photcalib["coeffs"])
    loc["coeffs_err"] = list(photcalib["coeffs_err"])

    if output_json != "":
        print("Dumping solution data into file -> ", output_json)
        with open(output_json, 'w') as json_file :
            json.dump(loc, json_file, sort_keys=False, indent=4)
    
    tbl = Table()    # set output table
    tbl.add_column(Column(data=np.array([]), name='gaiadr2_id'))
    skycoord_colnames = ['ra', 'dec', 'e_ra', 'e_dec']
    for col in skycoord_colnames :
        tbl.add_column(Column(data=np.array([]), name=col,unit=u.deg))
    pm_colnames = ['pmRA', 'pmDE']
    for col in pm_colnames :
        tbl.add_column(Column(data=np.array([]), name=col,unit=u.mas/u.yr))
    epoch_colname = 'epoch_yr'
    tbl.add_column(Column(data=np.array([]), name=epoch_colname,unit=u.yr))
    tbl.add_column(Column(data=np.array([]), name='time_bjd',unit=u.d))
    tbl.add_column(Column(data=np.array([]), name='airmass'))
    data_colnames = ['mag_g', 'emag_g',
                    'mag_r', 'emag_r',
                    'mag_i', 'emag_i',
                    'mag_z', 'emag_z',
                    'obs_mag_g', 'obs_emag_g',
                    'obs_mag_r', 'obs_emag_r',
                    'obs_mag_i', 'obs_emag_i',
                    'obs_mag_z', 'obs_emag_z']
    for col in data_colnames :
        tbl.add_column(Column(data=np.array([]), name=col, unit=u.mag))
        
    if photcalib["time_from_midnight"] is not None :
        tbl.add_column(Column(data=np.array([]), name='time_from_midnight'))

    for i in range(len(photcalib['time_bjd'])) :
    
        tbl.add_row()
        
        tbl['gaiadr2_id'][i] = photcalib['gaiadr2_id'][i]
        tbl['ra'][i] = photcalib['ra'][i]
        tbl['dec'][i] = photcalib['dec'][i]
        tbl['pmRA'][i] = photcalib['pmRA'][i]
        tbl['pmDE'][i] = photcalib['pmDE'][i]
        tbl['epoch_yr'][i] = photcalib['epoch_yr'][i]
        
        tbl['time_bjd'][i] = photcalib['time_bjd'][i]
        tbl['airmass'][i] = photcalib['airmass'][i]
        if photcalib['time_from_midnight'] is not None :
            tbl['time_from_midnight'][i] = photcalib['time_from_midnight'][i]

        for band in photcalib['bands'] :
            tbl['mag_{}'.format(band)][i] = photcalib['mag_{}'.format(band)][i]
            tbl['emag_{}'.format(band)][i] = photcalib['emag_{}'.format(band)][i]
            tbl['obs_mag_{}'.format(band)][i] = photcalib['obs_mag_{}'.format(band)][i]
            tbl['obs_emag_{}'.format(band)][i] = photcalib['obs_emag_{}'.format(band)][i]

    if output_csv != "" :
        print("Dumping clean catalog data into file -> ", output_csv)
        tbl.write(output_csv, overwrite=True)
    
    return loc, tbl
   
#likelihood function
def lnlikelihood(coeffs, obs_mags, obs_emags, mags, airmass, ncoeffs=3, nbands=4, time_from_midnight=None) :
    residuals, errors = errfunc(coeffs, obs_mags, obs_emags, mags, airmass, ncoeffs=ncoeffs, nbands=nbands, time_from_midnight=time_from_midnight, return_errors=True)
    sum_of_residuals = 0
    for i in range(len(residuals)) :
        sum_of_residuals += np.nansum(residuals[i]**2 + np.log(2.0 * np.pi * (errors[i] * errors[i])))
    ln_likelihood = -0.5 * (sum_of_residuals)
    return ln_likelihood

# prior probability from definitions in priorslib
def lnprior(theta):
    total_prior = 0.0
    """
    # below one may set bounds
    for i in range(len(theta)) :
        if -1e10 < theta[i] < +1e10:
            return 0
        return -np.inf
    """
    return total_prior

#posterior probability
def lnprob(theta, obs_mags, obs_emags, mags, airmass, ncoeffs=3, nbands=4, time_from_midnight=None):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnlike = lnlikelihood(theta, obs_mags, obs_emags, mags, airmass, ncoeffs=ncoeffs, nbands=nbands, time_from_midnight=time_from_midnight)
    prob = lp + lnlike
    if np.isfinite(prob) :
        return prob
    else :
        return -np.inf
    
    
#- Derive best-fit params and their 1-sigm a error bars
def best_fit_params(samples, use_mean=False, best_fit_from_mode=False, nbins = 30, plot_distributions=False, use_mean_error=True, verbose = False) :

    theta, theta_err = [], []
    
    if use_mean :
        npsamples = np.array(samples)
        values = []
        for i in range(len(samples[0])) :
            mean = np.mean(npsamples[:,i])
            err = np.std(npsamples[:,i])
            values.append([mean,err,err])
    else :
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))

        max_values = []
        
        for i in range(len(values)) :
            hist, bin_edges = np.histogram(samples[:,i], bins=nbins, range=(values[i][0]-5*values[i][1],values[i][0]+5*values[i][2]), density=True)
            xcen = (bin_edges[:-1] + bin_edges[1:])/2
            mode = xcen[np.argmax(hist)]
            max_values.append(mode)
                
            if plot_distributions :
                nmax = len(samples[:,i])
                plt.step(xcen, hist, where='mid')
                plt.vlines([values[i][0]], np.min(0), np.max(hist), ls="--", label="median")
                plt.vlines([mode], np.min(0), np.max(hist), ls=":", label="mode")
                plt.ylabel(r"Probability density",fontsize=18)
                plt.xlabel(r"coeff_{}".format(i),fontsize=18)
                plt.legend()
                plt.show()

                plt.plot(samples[:,i],label="coeff_{}".format(i), alpha=0.5, lw=0.5)
                plt.hlines([], np.min(0), np.max(nmax), ls=":", label="mode",zorder=2)
                plt.hlines([values[i][0]], np.min(0), np.max(nmax), ls="-", label="median",zorder=2)
                plt.ylabel(r"coeff_{}".format(i),fontsize=18)
                plt.xlabel(r"MCMC iteration",fontsize=18)
                plt.legend(fontsize=18)
                plt.show()
                    
        max_values = np.array(max_values)

    for i in range(len(values)) :
        if best_fit_from_mode :
            theta.append(max_values[i])
        else :
            theta.append(values[i][0])

        if use_mean_error :
            merr = (values[i][1]+values[i][2])/2
            theta_err.append(merr)
            if verbose :
                print("coeff_{} = {} +/- {}".format(i, values[i][0], merr))
        else :
            theta_err.append((values[i][1],values[i][2]))
            if verbose :
                print("coeff_{} = {} + {} - {}".format(i, values[i][0], values[i][1], values[i][2]))
            
    return theta, theta_err

    
def run_mcmc_fit(photcalib, amp=1e-4, nwalkers=32, niter=100, burnin=20, samples_filename="", appendsamples=False, verbose=False) :
    
    theta = photcalib["coeffs"]
    if verbose:
        print("initializing emcee sampler ...")

    #amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 50, 2000, 500
    ndim = len(theta)

    # Make sure the number of walkers is sufficient, and if not passing a new value
    if nwalkers < 2*ndim:
        nwalkers = 2*ndim
        print("WARNING: resetting number of MCMC walkers to {}".format(nwalkers))
    
    backend = None
    if samples_filename != "" :
        # Set up the backend
        backend = emcee.backends.HDFBackend(samples_filename)
        if appendsamples == False :
            backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [photcalib["obs_mags"], photcalib["obs_emags"] , photcalib["mags"], photcalib["airmass"], photcalib["ncoeffs"], photcalib["nbands"], photcalib["time_from_midnight"]], backend=backend)

    pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
    #--------

    #- run mcmc
    if verbose:
        print("Running MCMC ...")
        print("N_walkers=",nwalkers," ndim=",ndim)

    sampler.run_mcmc(pos, niter, progress=True)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
    #--------

    fig = corner.corner(samples, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], labelpad=2.0, labelsize=10, show_titles=False)
    plt.show()

    theta, theta_err = best_fit_params(samples, use_mean=False, best_fit_from_mode=True, nbins=30, plot_distributions=True, use_mean_error=True, verbose=True)
    
    photcalib["coeffs"] = theta
    photcalib["coeffs_err"] = theta_err
        
    return photcalib

    
parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data directory',type='string', default="")
parser.add_option("-s", "--input_solution", dest="input_solution", help='Input solution json file',type='string', default="")
parser.add_option("-o", "--output", dest="output", help='Output json solution',type='string', default="")
parser.add_option("-d", "--output_catalog", dest="output_catalog", help='Output master clean catalog',type='string', default="")
parser.add_option("-m", action="store_true", dest="mcmc", help="run mcmc", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with sparc4_photometric_calibration.py -h "); sys.exit(1);

# input data
inputdata = sorted(glob.glob(options.input))

# initialize photometric calibration container
photcalib = initialize_phot_calibration(inputdata[0], solution=options.input_solution, use_time_from_midnight=True)

# run fit to obatain the photometric calibration
photcalib = run_photometric_calibration(photcalib, n_sigma_clip=3, niter=8)

# run MCMC fit
if options.mcmc :
    photcalib = run_mcmc_fit(photcalib, amp=1e-4, nwalkers=32, niter=500, burnin=100, verbose=True)

# do plots
plot_phot_calib_results(photcalib)

# save solution
solution, tbl = save_solution(photcalib, output_json=options.output, output_csv=options.output_catalog)


# Todo:
# - improve plots


