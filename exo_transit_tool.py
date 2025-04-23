"""
    Created on Aug 23 2024
    
    Description: A tool to analyze transits of exoplanets observed with SPARC4
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:
    
    python exo_transit_tool.py --exoplanet="CoRoT-2 b" --object=CoRot-2 --input="/Users/eder/Science/Transits-OPD_2024A/CoRot-2/20240704/" --obj_indexes="0,3,3,4" --comps1="2,4,5,6" --comps2="0,1,2,4,5" --comps3="1,2,5,6,7" --comps4="2,3,6,7"  --planet_priors=myCoRoT-2b.pars
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os
from optparse import OptionParser

import sparc4.product_plots as s4plt
import sparc4.pipeline_lib as s4pipelib

import sparc4_analysis_lib as s4analyb
import sparc4_exoplanet_lib as s4exolib
import sparc4_analysis_plot_lib as s4analyplt
import sparc4_fit_lib as s4fitlib

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table

import priorslib
import fitlib
import gp_lib
from scipy.interpolate import interp1d
import exoplanetlib
from uncertainties import ufloat

import glob

exo_transit_tool_path = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data directory',type='string', default="./")
parser.add_option("-l", "--lcdata", dest="lcdata", help='Pattern for input light curve data',type='string',default="")
parser.add_option("-d", "--results_dir", dest="results_dir", help='Results directory path',type='string',default="./")
parser.add_option("-o", "--object", dest="object", help='Object ID',type='string', default="")
parser.add_option("-a", "--aperture_radius", dest="aperture_radius", help='Aperture radius in pixels',type='int', default=12)
parser.add_option("-e", "--exoplanet", dest="exoplanet", help='Exoplanet ID',type='string', default="")
parser.add_option("-r", "--planet_priors", dest="planet_priors", help='Planet priors file name',type='string',default="")
parser.add_option("-x", "--obj_indexes", dest="obj_indexes", help='Object indexes',type='string', default="")
parser.add_option("-1", "--comps1", dest="comps1", help='Comparisons indexes for channel 1',type='string', default="")
parser.add_option("-2", "--comps2", dest="comps2", help='Comparisons indexes for channel 1',type='string', default="")
parser.add_option("-3", "--comps3", dest="comps3", help='Comparisons indexes for channel 1',type='string', default="")
parser.add_option("-4", "--comps4", dest="comps4", help='Comparisons indexes for channel 1',type='string', default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with exo_transit_tool.py -h "); sys.exit(1);

calibrate_photometry = False
calibrate_astrometry = False

# read input lc data
inputlcdata = None
if options.lcdata != "" :
    inputlcdata = sorted(glob.glob(options.lcdata))


# load exoplanet catalog
exoplanet_catalog_path = os.path.join(exo_transit_tool_path,'data/exoplanet.eu_catalog.csv')

# initialize exoplanet from data base
exoplanet = s4exolib.get_exoplanet(options.exoplanet,exoplanet_catalog=exoplanet_catalog_path)

planet_priors_file = options.planet_priors

if exoplanet != {} :

    if planet_priors_file == "" :
        # define output planet prior filename
        planet_priors_file = "{}.pars".format(exoplanet['name'].replace(" ",""))

        # create planet prior file from exoplanet catalog parameters
        s4exolib.create_planet_prior_file(exoplanet, planet_priors_file, hasrvdata=False, circular_orbit=True, set_star_density=False, use_impact_parameter=False, planet_index=0, n_planets=1, all_parameters_fixed=False)

    # get star name for source identification
    star_name = exoplanet['star_name']
    
    # reset object name to star's name to optimize matching
    options.object = star_name
    
    # find Simbad match for main object
    exoplanet_match_simbad, exoplanet_coords = s4analyb.match_object_with_simbad(star_name, ra=exoplanet['ra'], dec=exoplanet['dec'], search_radius_arcsec=10)

else :
    star_name = options.object
    
    # find Simbad match for main object
    exoplanet_match_simbad, exoplanet_coords = s4analyb.match_object_with_simbad(star_name, search_radius_arcsec=10)
    
print("planet_priors_file=",planet_priors_file)


# input data directory
datadirs = options.input.split(",")

# set sparc4 data bundle to store data from several nights
s4bundle = {}

for datadir in datadirs :

    # initialize dict to store data for current key defined as the night dir name
    s4bundle[datadir] = {}
    
    # load SPARC4 data products in the directory
    s4products = s4analyb.load_data_products(datadir, object_id=options.object)

    objprods={}
    for key in s4products.keys() :
        objprods = s4products[key]
        print("Mode (key) : {}".format(key))
        for ch in range(4) :
            print("******* CHANNEL {} ***********".format(ch+1))
            print(objprods["object"], "STACK: ", objprods["stack"][ch])
            print(objprods["object"], "POLAR L2: ", objprods["polarl2"][ch])
            print(objprods["object"], "POLAR L4: ", objprods["polarl4"][ch])
            print(objprods["object"], "TS: ", objprods["ts"][ch])
            print(objprods["object"], "LC: ", objprods["lc"][ch])
            print("\n")
        # break to get only the first set of files
        break
    
    # - recalibrate astrometry -- not needed if product has a good enough astrometry
    if calibrate_astrometry :
        catalogs, objprods = s4analyb.calibrate_astrometry_in_stack_catalog(objprods, use_catalog=False, aperture_radius=options.aperture_radius, niter=3, plot=options.plot)

    object_indexes = []
    if options.obj_indexes == "" :
        #  identify main source in the observed catalogs
        object_indexes = s4analyb.get_object_indexes_in_catalogs(objprods, obj_id=options.object, simbad_id=exoplanet_match_simbad["MAIN_ID"][0])
    else :
        objidx = options.obj_indexes.split(",")
        for i in range(len(objidx)) :
            object_indexes.append(int(objidx[i]))
    print(object_indexes)

    #  calibrate photometry to a standard system
    if calibrate_photometry :
        catalogs, objprods = s4analyb.calibrate_photometry_in_stack_catalog(objprods, aperture_radius=options.aperture_radius, vizier_catalog_name='SkyMapper', plot=options.plot)
        for ch in range(4) :
            print("Band {}: (a,b)={}".format(ch+1,objprods["phot_calibration"][ch]))

    master_catalog = s4analyb.create_master_catalog()
    #try :
    #    # create a master catalog matching sources for all 4 channels
    #    master_catalog = s4analyb.master_catalog_in_four_bands(objprods, calibrate_photometry=calibrate_photometry)
    #except :
    #    print("Could not populate master catalog")
    
    comps = s4analyb.select_comparison_stars(master_catalog, target_indexes=object_indexes, comps1=options.comps1, comps2=options.comps2, comps3=options.comps3, comps4=options.comps4, max_mag_diff=2)

    if options.plot :
        s4analyplt.plot_s4data_for_diffphot(objprods, object_indexes, comps, aperture_radius=options.aperture_radius, object_name=options.object)
    for ch in range(4) :
        print("Channel {} -> Target = {} ; Comparisons: {}".format(ch+1, object_indexes[ch],comps[ch]))

    lcs = s4analyb.differential_light_curves(objprods, object_indexes, comps, aperture_radius=options.aperture_radius, binsize=0.005, object_name=options.object, plot=options.plot)

    wppositions = [None, None, None, None]
    if objprods["instmode"] == 'POLAR' :
        plot_total_polarization = False
    
        tss = s4analyb.get_polarimetry_time_series(objprods, object_indexes, comps, aperture_radius=options.aperture_radius, binsize=0.002, object_name=options.object, plot_total_polarization=plot_total_polarization, plot=options.plot)
        tss = s4analyb.get_polarimetry_time_series(objprods, object_indexes, comps, aperture_radius=options.aperture_radius, binsize=0.002, plot_theta=True, plot_u=True, plot_total_polarization=plot_total_polarization, object_name=options.object, plot=options.plot)
        for ch in range(4) :
            print(objprods["lc"][ch])
            catalog_ext = "CATALOG_PHOT_AP{:03d}".format(options.aperture_radius)
            wppositions[ch] = s4analyb.get_wppos(objprods["lc"][ch], target=object_indexes[ch], comps=comps[ch], extname=catalog_ext,   verbose=False)
        # store data that will be used later
        
    s4bundle[datadir]['objprods'] = objprods
    s4bundle[datadir]['master_catalog'] = master_catalog
    s4bundle[datadir]['object_indexes'] = object_indexes
    s4bundle[datadir]['comps'] = comps
    s4bundle[datadir]['lcs'] = lcs
    s4bundle[datadir]['wppositions'] = wppositions

use_tess_data = True
use_linear_ld = False

walkers, burnin, nsteps = 52, 100, 300

times, fluxes, fluxerrs, instrument_indexes = [], [], [], []

if use_tess_data :
    # Get TESS data, and fit transits
    tess_fit = s4fitlib.fit_tess_data(data_files=inputlcdata, planet_priors_file=options.planet_priors, object_name=options.object, planet_index=0, calib_polyorder=2, ols_fit=True, mcmc_fit=True, walkers=walkers, nsteps=nsteps, burnin=burnin, priors_dir=options.results_dir, use_linear_ld=use_linear_ld, best_fit_from_mode=False, plot=options.plot, lc_in_txt_format=False, transit_window_size=2)
   
    # fit transits - HATS-9 using K2 data
    #tess_fit = s4fitlib.fit_tess_data(data_files=inputlcdata, planet_priors_file=options.planet_priors, object_name=options.object, planet_index=0, calib_polyorder=2, ols_fit=True, mcmc_fit=True, walkers=walkers, nsteps=nsteps, burnin=burnin, priors_dir=options.results_dir, use_linear_ld=use_linear_ld, best_fit_from_mode=False, plot=options.plot, lc_in_txt_format=True, transit_window_size=5)
    
    # Run global fit using TESS data and TESS preliminary fit as prior
    planet_priors_file = priorslib.update_prior_initial_guess_from_posterior(tess_fit["planet_priors_file"], tess_fit["planet_posteriors_file"])
    
    times, fluxes, fluxerrs, instrument_indexes = tess_fit["times"], tess_fit["fluxes"], tess_fit["fluxerrs"], tess_fit["instrument_indexes"]
  
for datadir in datadirs :
    # Get SPARC4 data from each night
    times, fluxes, fluxerrs, instrument_indexes = s4analyb.get_fluxes_from_lcs(s4bundle[datadir]['lcs'], times=times, fluxes=fluxes, fluxerrs=fluxerrs, instrument_indexes=instrument_indexes)

# run a global fit using TESS and SPARC4 data together
results = s4fitlib.fit_transits(planet_priors_file, times, fluxes, fluxerrs, instrument_indexes, object_name=options.object, planet_index=0, calib_polyorder=1, ols_fit=True, mcmc_fit=True, walkers=walkers, nsteps=nsteps, burnin=burnin, priors_dir=options.results_dir, detrend_data=True, applySgimaClip=True, niter=2, binsize=0.01, best_fit_from_mode=True, plot_detrend=False, plot=options.plot)

planet_priors = results["posterior"]["planet_posterior_file"]
print("*** Posteriors file from all data: {}".format(planet_priors))

times, fluxes, fluxerrs = results["times"], results["fluxes"], results["fluxerrs"]

instidx = np.array(instrument_indexes)
dataset_idx = np.arange(len(instidx))

timesampling=0.001
t0,tf = 1e50,0
for ch in range(1,5) :
    keep = (instidx == ch)
    for idx in dataset_idx[keep] :
        if times[idx][0] < t0 :
            t0 = times[idx][0]
        if times[idx][-1] > tf :
            tf = times[idx][-1]
time = np.arange(t0, tf, timesampling)

tbl, tbl_params = Table(), Table()
tbl['TIME'] = time

planet_index = 0

wl = np.array([464.,614.,754.,892.])
wlerr = np.array([132.,95.,101.,119.])

rp, rperr = np.array([]), np.array([])
u0, u0err = np.array([]), np.array([])
u1, u1err = np.array([]), np.array([])

s4times = []
s4fluxes, s4fluxerrs = [], []
s4posteriors = []

for ch in range(4) :
    keep = (instidx == ch+1)
    loctimes, locfluxes, locfluxerrs = [], [], []
    locinstrument_indexes = []
    
    for idx in dataset_idx[keep] :
        loctimes.append(times[idx])
        locfluxes.append(fluxes[idx])
        locfluxerrs.append(fluxerrs[idx])
        locinstrument_indexes.append(instrument_indexes[idx])
      
    ch_planet_priors = planet_priors.replace(".pars","_ch{}.pars".format(ch+1))
    
    # Run global fit using TESS data and TESS preliminary fit as prior
    planet_priors_file = priorslib.update_posteriors_file(planet_priors, keys_to_update=["per_000","rp_000","u0_000","u1_000"], types=["FIXED","Uniform","Uniform","Uniform"], values=[None,None,None,None], values_l=[None,0.,0.,0.], values_u=[None,1.,3.,3.], updated_posteriors_file=ch_planet_priors)
          
    # Now run a fit to a single channel of SPARC4
    ch_results = s4fitlib.fit_transits(ch_planet_priors, loctimes, locfluxes, locfluxerrs, locinstrument_indexes, object_name=options.object, planet_index=0, calib_polyorder=1, ols_fit=True, mcmc_fit=True, walkers=walkers, nsteps=nsteps, burnin=burnin, priors_dir="./", detrend_data=True, applySgimaClip=False, niter=1, binsize=0.01, best_fit_from_mode=True, plot_detrend=False, plot=True)
    
    if len(datadirs) > 1 :
        continue
    
    # E.M. I commented this part because the wppositions array does not match the flux array
    if objprods["instmode"] == 'POLAR' :
        locfluxes, locfluxerrs = s4fitlib.remove_waveplate_modulation(loctimes, locfluxes, locfluxerrs, wppositions[ch], ch_results["posterior"], combine_by_median=True, plot=True, verbose=True)

        # Now run a fit to a single channel of SPARC4
        ch_results = s4fitlib.fit_transits(ch_planet_priors, loctimes, locfluxes, locfluxerrs, locinstrument_indexes, object_name=options.object, planet_index=0, calib_polyorder=1, ols_fit=True, mcmc_fit=True, walkers=walkers, nsteps=nsteps, burnin=burnin, priors_dir="./", detrend_data=True, applySgimaClip=False, niter=1, binsize=0.01, best_fit_from_mode=True, plot_detrend=False, plot=False)

    ch_tbl = s4fitlib.diff_light_curve(ch_results["times"], ch_results["fluxes"], ch_results["fluxerrs"], ch_results["posterior"], nsig=100, model_time_sampling=0.001, offset_nsig=10, combine_by_median=False, binsize = 0.005, output="", instrum_index=0, use_calibs=True, plot_comps=False, plot=False, verbose=True)

    bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(ch_tbl["TIME"], ch_tbl["FLUX"], ch_tbl["FLUXERR"], median=True, binsize=0.003)
    
    #f = interp1d(bin_time, bin_flux, kind='cubic',bounds_error=False,)
    y, yerr = gp_lib.interp_gp(time, bin_time, bin_flux, bin_fluxerr, [[ch_tbl["TIME"][0],ch_tbl["TIME"][-1]]], verbose=False, plot=False)
    
    plt.plot(ch_tbl["TIME"],ch_tbl["FLUX"],'k.',alpha=0.3)
    plt.errorbar(bin_time,bin_flux,yerr=bin_fluxerr,fmt='ko',alpha=0.7)
    plt.plot(ch_tbl["TIME"],ch_tbl["TRANSIT_MODEL"],'-',color='darkgreen',lw=3)
    plt.plot(time,y,'-',color="#ff7f0e",lw=1.5)
    plt.fill_between(time, y+yerr, y-yerr, color="#ff7f0e", alpha=0.3, edgecolor="none")

    posterior = ch_results["posterior"]
    planet_params = posterior["planet_params"]

    rp = np.append(rp,planet_params['rp_{0:03d}'.format(planet_index)])
    rp_err = (planet_params['rp_{0:03d}_err'.format(planet_index)][0]+planet_params['rp_{0:03d}_err'.format(planet_index)][1])/2
    rperr = np.append(rperr,rp_err)

    u0_err = (planet_params['u0_{0:03d}_err'.format(planet_index)][0]+planet_params['u0_{0:03d}_err'.format(planet_index)][1])/2
    u0 = np.append(u0,planet_params['u0_{0:03d}'.format(planet_index)])
    u0err = np.append(u0err,u0_err)
    
    if not use_linear_ld :
        u1_err = (planet_params['u1_{0:03d}_err'.format(planet_index)][0]+planet_params['u1_{0:03d}_err'.format(planet_index)][1])/2
        u1 = np.append(u1,planet_params['u1_{0:03d}'.format(planet_index)])
        u1err = np.append(u1err,u1_err)
    
    transit_model = np.full_like(time, 1.0)
    for j in range(int(posterior["n_planets"])) :
        planet_transit_id = "{0}_{1:03d}".format('transit', j)
        if planet_params[planet_transit_id] :
            transit_model *= exoplanetlib.batman_transit_model(time, planet_params, planet_index=j)
            
    plt.xlabel("Time (BJD)",fontsize=20)
    plt.ylabel(r"Relative flux",fontsize=20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.minorticks_on()
    plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
    plt.show()

    tbl['FLUX_CH{}'.format(ch+1)] = y
    tbl['FLUXERR_CH{}'.format(ch+1)] = yerr
    tbl['TRANSIT_MODEL_CH{}'.format(ch+1)] = transit_model

    s4times.append(np.array(ch_tbl["TIME"],dtype=float))
    s4fluxes.append(np.array(ch_tbl["FLUX"],dtype=float))
    s4fluxerrs.append(np.array(ch_tbl["FLUXERR"],dtype=float))
    s4posteriors.append(posterior)


# plot final data and model for all 4 channels
s4fitlib.plot_s4_transit_light_curves(s4times, s4fluxes, s4fluxerrs, s4posteriors, model_time_sampling=0.001, object_name=options.object)

tbl_params["WAVELENGTH"] = wl
tbl_params["WAVELENGTH_ERR"] = wlerr
tbl_params["PLANET_RADIUS"] = rp
tbl_params["PLANET_RADIUS_ERR"] = rperr
tbl_params["LDCOEFF0"] = u0
tbl_params["LDCOEFF0_ERR"] = u0err
if not use_linear_ld :
    tbl_params["LDCOEFF1"] = u1
    tbl_params["LDCOEFF1_ERR"] = u1err

plt.errorbar(wl,rp,xerr=wlerr,yerr=rperr,fmt="o",color='k')
plt.xlabel("wavelength (nm)",fontsize=20)
plt.ylabel(r"R$_p$/R$_\star$",fontsize=20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.minorticks_on()
plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
plt.show()

bands = ['g','r','i','z']

for ch1 in [1,2,3,4] :
    for ch2 in [1,2,3,4] :
        if ch1 != ch2 and ch1 < ch2 and 'FLUX_CH{}'.format(ch1) in tbl.colnames and 'FLUX_CH{}'.format(ch2) in tbl.colnames:
        
            model1 = tbl['TRANSIT_MODEL_CH{}'.format(ch1)]
            flux1 = tbl['FLUX_CH{}'.format(ch1)]
            fluxerr1 = tbl['FLUXERR_CH{}'.format(ch1)]
            
            model2 = tbl['TRANSIT_MODEL_CH{}'.format(ch2)]
            flux2 = tbl['FLUX_CH{}'.format(ch2)]
            fluxerr2 = tbl['FLUXERR_CH{}'.format(ch2)]
            
            ym, y, yerr = s4fitlib.color_light_curve(tbl['TIME'], model1, flux1, fluxerr1, model2, flux2, fluxerr2, label1="{}-band".format(bands[ch1-1]),label2="{}-band".format(bands[ch2-1]), title="Color light curve: {}-band / {}-band".format(bands[ch1-1],bands[ch2-1]), plot=True)
            
            tbl['TRMODELRATIO_CH{}_OVER_CH{}'.format(ch1,ch2)] = ym
            tbl['FLUXRATIO_CH{}_OVER_CH{}'.format(ch1,ch2)] = y
            tbl['FLUXRATIOERR_CH{}_OVER_CH{}'.format(ch1,ch2)] = yerr

tbl.write("{}/{}_lightcurve_transit_model.csv".format(options.results_dir,options.object.replace(" ","")), overwrite=True)
tbl_params.write("{}/{}_rp_and_ldcoeffs.csv".format(options.results_dir,options.object.replace(" ","")), overwrite=True)

###
# 1. fit all data together using tess priors - detrend data and remove outliers at this point -- save detrend params
# 2. fit each channel's data independently using global posterior as prior
# 3. compile all results from the fit into one table - include number of comparisons, total magnitude, rms, etc..
# 4. Build color light curves - export as csv tables

# For each stack image do :
#   - select potential comparisons based on magnitudes, color differences.
##   - match selection for all bands
#   - plot stack image, tag main target and selected comparisons

# For light curve files in all bands :
#   - Read in all flux data and write them to a global photometric calibration file
#   - Calibrate photometry to a standard system
#   - measure snr, rms of individual stars
#   - measure differential photometry of each selected comparison with respect to all combinations of the other comparisons, excluding the target
#   - select the most stable comparisons and tag those that could be variable stars
#   - measure differential photometry for the target with respect to each comparison
#   - global fit for all comparison's light curves in all bands, including a global transit model, with independent LDCs for each band and a calibration polynomial for each curve. Include TESS data, and any other photometry data for this fit. In the case of polarimetry, include an additional calibration per WPPOS.
#   - save calibration coefficients and clean the data, leaving only clean photometry and the transit models.
#   - combine photometry from all comparisons in each band, and bin data.
#   - plot light curves of all bands
#   - fit GP models to interpolate data in each band and calculate color light curves
#   -
