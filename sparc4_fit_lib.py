"""
    Created on Aug 23 2024

    Description: Library of recipes for the fit analysis of SPARC4 data

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import priorslib
import fitlib
import tess
import exoplanetlib
from scipy import optimize
from scipy import stats
from astropy.table import Table
from uncertainties import ufloat
from astropy.io import ascii

def fit_transits(planet_priors_file, times, fluxes, fluxerrs, instrument_indexes, object_name="", planet_index=0, calib_polyorder=1, ols_fit=False, mcmc_fit=False, walkers=32, nsteps=500, burnin=100, amp=1e-4, priors_dir="./", detrend_data=False, applySgimaClip=False, niter=1, binsize=0.01,  best_fit_from_mode=False, plot_detrend=False, plot=False) :
      
    loc = {}
    
    posterior = None
    
    priors = fitlib.read_transit_rv_priors(planet_priors_file, 0, len(times), planet_index=planet_index, calib_polyorder=calib_polyorder,verbose=False)
    
    # Fit calibration parameters for initial guess
    if ols_fit :
        posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")
    else :
        posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED")

    for i in range(niter) :
        if ols_fit :
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
        if detrend_data :
            fluxes, fluxerrs = remove_systematics(times, fluxes, fluxerrs, posterior, calib_polyorder=5, binsize=binsize, plot=plot_detrend)
        if applySgimaClip :
            times, fluxes, fluxerrs = clean_data(times, fluxes, fluxerrs, posterior, n_sigma_clip=4., plot=False, verbose=False)

    if detrend_data :
        # Fit calibration parameters for initial guess
        if ols_fit :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")
            posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=False)
        else :
            posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED")

    if plot :
        # plot light curves and models in priors
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

    if mcmc_fit :
        # Final fit with MCMC
        # Make sure the number of walkers is sufficient, and if not assing a new value
        if walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            walkers = 2*len(posterior["theta"])

        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=amp, nwalkers=walkers, niter=nsteps, burnin=burnin, verbose=True, plot=plot, best_fit_from_mode=best_fit_from_mode, appendsamples=False, plot_individual_transits=False)
    else :
    
        if ols_fit :
            # below the ideal would be to update values from OLS fit -> to do
            #planet_params = posterior["planet_params"]
            # Run global fit using TESS data and TESS preliminary fit as prior
            #planet_priors_file = priorslib.update_posteriors_file(planet_priors_file, keys_to_update=["rp_000","u0_000","u1_000"], types=["Uniform","Uniform","Uniform"], values=[None,None,None], values_l=[0.,0.,0.], values_u=[1.,3.,3.], updated_posteriors_file=ch_planet_priors)
    
            posterior["planet_posterior_file"] = planet_priors_file
        
    loc["posterior"] = posterior
    loc["times"] = times
    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs
    
    return loc


def read_tess_data_from_files(filelist) :

    time = np.array([])
    relflux, fluxerr = np.array([]), np.array([])
    phase, model = np.array([]), np.array([])
    #[('TIME', '>f8'), ('TIMECORR', '>f4'), ('CADENCENO', '>i4'), ('PHASE', '>f4'), ('LC_INIT', '>f4'), ('LC_INIT_ERR', '>f4'), ('LC_WHITE', '>f4'), ('LC_DETREND', '>f4'), ('MODEL_INIT', '>f4'), ('MODEL_WHITE', '>f4')]))

    for filename in filelist :
        data = fits.getdata(filename, 1)
        keep = np.isfinite(data['LC_DETREND']) *  np.isfinite(data['LC_INIT_ERR'])
        if convert_times_to_bjd :
            time = np.append(time,data['TIME'][keep])
        else :
            time = np.append(time,data['TIME'][keep]+2457000)
        relflux = np.append(relflux,data['LC_DETREND'][keep])
        fluxerr = np.append(fluxerr,data['LC_INIT_ERR'][keep])
        model = np.append(model,data['MODEL_INIT'][keep])
        phase = np.append(phase,data['PHASE'][keep])

    if plot :
        # Plot the detrended photometric time series
        plot_detrended_timeseries(time, relflux, fluxerr, model, star_name=object)
        plot_folded_light_curve (selected_dvt_files)

    sorted = np.argsort(time)
    loc["object_name"] = object_name
    loc["time"] = time[sorted]
    loc["flux"] = relflux[sorted]
    loc["fluxerr"] = fluxerr[sorted]
    loc["model"] = model[sorted]
    loc["phase"] = phase[sorted]
    loc['nflux'] = loc["flux"]
    loc['nfluxerr'] = loc["fluxerr"]




def fit_tess_data(data_files=None, planet_priors_file="", object_name="", planet_index=0, inst_index=0, calib_polyorder=1, ols_fit=False, mcmc_fit=False, walkers=32, nsteps=500, burnin=100, amp=1e-6, priors_dir="./", use_linear_ld=True, best_fit_from_mode=False, plot=False, verbose=False, lc_in_txt_format=False, t0=2454833., tepoch=2469.1764, period=1.9153073, tdur=5, transit_window_size=2) :

    loc = {}
    
    if lc_in_txt_format :
        
        times, fluxes, fluxerrs = [], [], []
        tdur_days = tdur/24.
        twindow = transit_window_size * tdur_days
        
        for j in range(len(data_files)) :
            
            tbl = ascii.read(data_files[j], data_start=1)
            
            t, f = np.array(tbl['col1'],dtype=float), np.array(tbl['col2'],dtype=float)
            ef = f/10000 # guess k2 error
            
            planet_tcs = []
            
            # calcualte central time of transits
            tcs = tess.calculate_tcs_within_range(np.nanmin(t), np.nanmax(t), tepoch, period)
            for i in range(len(tcs)) :
            
                t1 = tcs[i] - twindow/2
                t2 = tcs[i] + twindow/2

                window = (t > t1) & (t < t2) & (np.isfinite(f)) & (np.isfinite(t))
                
                print(i, tcs[i], t1, t2, len(t[window]))
                if len(t[window]) > 5 :
                    window_size = t[window][-1] - t[window][0]
                    if window_size > tdur_days*2 :
                        times.append(t[window]+t0)
                        fluxes.append(f[window])
                        fluxerrs.append(ef[window])
                    plt.plot(t[window]+t0,f[window],'.')
        plt.show()
            
    else :
        if data_files is None or planet_priors_file == "":
            # Download TESS DVT products and return a list of input data files
            dvt_filenames = tess.retrieve_tess_data_files(object_name, products_wanted_keys=["DVT"], verbose=True)

            # load TESS data from dvt files
            tesslc = tess.load_dvt_files(object_name, priors_dir=priors_dir, save_priors=True, use_star_density=False, use_impact_parameter=False, convert_times_to_bjd=True, plot=plot, use_linear_ld=use_linear_ld, verbose=True)
            
            if planet_priors_file == "" :
                planet_priors_file = tesslc["PRIOR_FILE"]
            
        if data_files is not None :
            min_npoints_per_bin = 3
            min_npoints_within_transit = 10
            binsize = 0.1
            
            tesslc = tess.load_lc(data_files, object_name=object_name, transit_window_size=transit_window_size, min_npoints_within_transit=min_npoints_within_transit, binbymedian=False, binsize=binsize, min_npoints_per_bin=min_npoints_per_bin, plot=plot, verbose=verbose, convert_times_to_bjd=True)
        
        planet = tesslc["PLANETS"][planet_index]
                        
        planet = tess.redefine_tessranges(planet, tesslc, planet_priors_file, planet_index=planet_index, transit_window_size=transit_window_size, verbose=True, plot=True)

        times, fluxes, fluxerrs = planet["times"], planet["fluxes"], planet["fluxerrs"]

    # initialize posterior
    posterior = None
    
    # read priors
    priors = fitlib.read_transit_rv_priors(planet_priors_file, 0, len(times), planet_index=planet_index, calib_polyorder=calib_polyorder,verbose=False)
    
    # Fit calibration parameters for initial guess
    if ols_fit :
        posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="Normal")
    else :
        posterior = fitlib.guess_calib(priors, times, fluxes, prior_type="FIXED")

    if ols_fit :
        # OLS fit involving all priors
        posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="Uniform", calib_unc=0.01, verbose=False, plot=plot)
        posterior = fitlib.fitTransits_ols(times, fluxes, fluxerrs, posterior, calib_post_type="FIXED", verbose=False, plot=plot)
        
    if plot :
        # plot light curves and models in priors
        fitlib.plot_mosaic_of_lightcurves(times, fluxes, fluxerrs, posterior)

    if mcmc_fit :
        # Final fit with MCMC
        # Make sure the number of walkers is sufficient, and if not assing a new value
        if walkers < 2*len(posterior["theta"]):
            print("WARNING: insufficient number of MCMC walkers, resetting nwalkers={}".format(2*len(posterior["theta"])))
            walkers = 2*len(posterior["theta"])

        posterior = fitlib.fitTransitsWithMCMC(times, fluxes, fluxerrs, posterior, amp=amp, nwalkers=walkers, niter=nsteps, burnin=burnin, verbose=True, plot=plot, best_fit_from_mode=best_fit_from_mode, appendsamples=False, plot_individual_transits=False)
     
    instrument_indexes = []
    for i in range(len(times)) :
        instrument_indexes.append(inst_index)
     
    loc["planet_priors_file"] = planet_priors_file
    loc["planet_posteriors_file"] = posterior["planet_posterior_file"]
    loc["posterior"] = posterior
    loc["times"] = times
    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs
    loc["instrument_indexes"] = instrument_indexes

    return loc


def clean_data(times, fluxes, fluxerrs, posterior, n_sigma_clip=5, plot=False, verbose=False) :

    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    out_fluxes, out_fluxerrs = [], []

    for i in range(len(times)) :
    
        transit_models = np.full_like(times[i], 1.0)

        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
            
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
                #plt.plot(times[i], fluxes[i], 'r.')
                #plt.plot(times[i], transit_models, 'g-', lw=2)
                #plt.show()

        keep = np.isfinite(fluxes[i]) *  np.isfinite(fluxerrs[i])
        
        flux_without_transit = fluxes[i] / transit_models

        median = np.nanmedian(flux_without_transit)
        mad = stats.median_abs_deviation(flux_without_transit[keep], scale="normal")

        if verbose :
            print("Comparison {} -> sigma = {:.10f}".format(i, mad))
        
        keep &= np.abs(flux_without_transit - median) < n_sigma_clip * mad
        keep &= fluxerrs[i] < 3 * mad
        
        if plot :
            plt.plot(times[i], flux_without_transit, 'g.')
            plt.plot(times[i][~keep], flux_without_transit[~keep], 'ro', alpha=0.3)
            plt.plot(times[i], np.full_like(times[i], median), 'b-')
            plt.plot(times[i], np.full_like(times[i], median + n_sigma_clip * mad), 'b:')
            plt.plot(times[i], np.full_like(times[i], median - n_sigma_clip * mad), 'b:')
            plt.show()

        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)
        out_flux[keep] = fluxes[i][keep]
        out_fluxerr[keep] = fluxerrs[i][keep]

        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)

    return times, out_fluxes, out_fluxerrs


def errfunc(coeffs, t, flux, fluxerr) :
    p = np.poly1d(coeffs)
    flux_model = p(t)
    residuals = (flux - flux_model) / fluxerr
    return residuals
    
def remove_systematics(times, fluxes, fluxerrs, posterior,  calib_polyorder=1, binsize=0.005, plot=False, verbose=False) :

    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    out_fluxes, out_fluxerrs = [], []

    for i in range(len(times)) :

        mtime = np.nanmedian(times[i])
        mflux = np.nanmedian(fluxes[i])

        transit_models = np.full_like(times[i], 1.0)
        
        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
        
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
                #plt.plot(times[i], fluxes[i], '.', color="grey",alpha=0.2)
                #plt.plot(times[i], transit_models, 'g-', lw=2)
                #plt.show()

        keep = np.isfinite(fluxes[i]) *  np.isfinite(fluxerrs[i])
        
        flux_without_transit = fluxes[i] / transit_models
        fluxerr_without_transit = fluxerrs[i] / transit_models

        bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(times[i][keep]-mtime, flux_without_transit[keep]-mflux, fluxerr_without_transit[keep], median=False, binsize=binsize)

        init_coeffs = np.array([])
        for j in range(calib_polyorder) :
            init_coeffs = np.append(init_coeffs,-0.1)
        init_coeffs[-1] = np.nanmean(flux_without_transit)

        if len(bin_time) > len(init_coeffs) + 2 :
            coeffs, success = optimize.leastsq(errfunc, init_coeffs, args=(bin_time, bin_flux, bin_fluxerr))
        else :
            coeffs = init_coeffs
            
        if plot :
            inst_label = ""
            if instrument_indexes is not None:
                inst_label = instrument_indexes[i]
            plt.title("Systematics for comparison star C{:03d} Instrum. idx: {}".format(i,inst_label), fontsize=18)
            plt.plot(times[i], flux_without_transit, 'g.',alpha=0.2)
            plt.errorbar(bin_time + mtime, bin_flux+mflux, yerr=bin_fluxerr, fmt='ko')
            plt.plot(times[i], np.poly1d(coeffs)(times[i]-mtime)+mflux, 'b-')
            plt.xlabel(r"time (BTJD)", fontsize=16)
            plt.ylabel(r"Flux ratio / transit model", fontsize=16)
            plt.show()

        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)
        out_flux[keep] = (fluxes[i][keep] / (np.poly1d(coeffs)(times[i][keep]-mtime)+mflux)) * mflux
        out_fluxerr[keep] = (fluxerrs[i][keep] / (np.poly1d(coeffs)(times[i][keep]-mtime)+mflux)) * mflux

        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)

    return out_fluxes, out_fluxerrs
    

def remove_waveplate_modulation(times, fluxes, fluxerrs, wppositions, posterior, combine_by_median=False, plot=True, verbose=False) :

    # get planet parameters from posterior
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    
    # initialize output data containers
    out_fluxes, out_fluxerrs = [], []

    # initialize temporary data containers
    fluxes_tmp, fluxerrs_tmp = [], []
    mtransits, mfluxes = [], []
    
    ref_time = times[0]
    median_delta_time = np.nanmedian(np.abs(ref_time[1:] - ref_time[:-1]))
    max_delta_time = median_delta_time * 16
    
    # find out sequences from data for first comparison
    wppos = wppositions[0]
    
    # loop over each lightcurve
    for i in range(len(times)) :
    
        # calculate transit models
        transit_models = np.full_like(times[i], 1.0)
        instrum_index = 0
        if instrum_params is not None and instrument_indexes is not None:
            instrum_index = instrument_indexes[i]
            
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(times[i], planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
        # store transit models to return it back to the data later
        mtransits.append(transit_models)
        
        # remove transits
        flux_without_transit = fluxes[i] / transit_models
        fluxerr_without_transit = fluxerrs[i] / transit_models
        
        # calculate mean flux without transit
        mflux = np.nanmedian(flux_without_transit)
        # store mean flux
        mfluxes.append(mflux)
        
        if plot :
            plt.plot(wppositions[i], flux_without_transit / mflux, ".", alpha=0.2)
            
        # store temporary fluxes with both transits and mean values removed
        fluxes_tmp.append(flux_without_transit / mflux)
        fluxerrs_tmp.append(fluxerr_without_transit / mflux)
   
    # cast to numpy arrays
    fluxes_tmp = np.array(fluxes_tmp)
    fluxerrs_tmp = np.array(fluxerrs_tmp)

    wppos_flux, wppos_fluxerr = [], []
    WAVEPLATEPOS = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    
    for k in range(len(WAVEPLATEPOS)) :
        keep = wppos == WAVEPLATEPOS[k]
        
        lc_mean = np.nan
        elc_mean = np.nan
        
        if len(wppos[keep]) :
            flat_fluxes_tmp = fluxes_tmp[:,keep].flatten()
            flat_fluxerrs_tmp = fluxerrs_tmp[:,keep].flatten()
        
            # calculate mean from all comparison stars
            lc_mean = np.average(flat_fluxes_tmp, weights=1./(flat_fluxerrs_tmp*flat_fluxerrs_tmp))
            elc_mean = np.nanstd(flat_fluxes_tmp-lc_mean)
            if combine_by_median :
                lc_mean = np.nanmedian(flat_fluxes_tmp, axis=0)
                elc_mean = np.nanmedian(np.abs(flat_fluxes_tmp-lc_mean), axis=0) / 0.67449
            
        wppos_flux.append(lc_mean)
        wppos_fluxerr.append(elc_mean)
    
    wppos_flux = np.array(wppos_flux)
    wppos_fluxerr = np.array(wppos_fluxerr)

    if plot :
        plt.errorbar(WAVEPLATEPOS, wppos_flux, yerr=wppos_fluxerr, fmt="ko")
        
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.minorticks_on()
        plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
                        
        plt.xlabel("waveplate position", fontsize=18)
        plt.ylabel("Mean flux", fontsize=18)
        plt.show()
    
    ffluxes_tmp = fluxes_tmp.flatten()
    # calculate statistics before gp correction to check efficiency
    sig_before_gp, mad_before_gp = np.nanstd(ffluxes_tmp), np.nanmedian(np.abs(ffluxes_tmp-np.nanmedian(ffluxes_tmp))) / 0.67449
  
    # loop over each comparison star to apply master GP correction to all lightcurves
    for i in range(len(times)) :
        out_flux, out_fluxerr = np.full_like(times[i],np.nan),np.full_like(times[i],np.nan)

        for k in range(len(WAVEPLATEPOS)) :
            keep = wppositions[i] == WAVEPLATEPOS[k]
            if len(wppositions[i][keep]) :
                # apply gp correction and recover transit signal and original median flux level
                fluxes_tmp[i][keep]  /= wppos_flux[k]
                fluxerrs_tmp[i][keep]  /= wppos_flux[k]
                
                out_flux[keep] = fluxes_tmp[i][keep] * mfluxes[i] * mtransits[i][keep]
                out_fluxerr[keep] = fluxerrs_tmp[i][keep] * mfluxes[i] * mtransits[i][keep]
                
        # store output fluxes
        out_fluxes.append(out_flux)
        out_fluxerrs.append(out_fluxerr)
    
    ffluxes_tmp = fluxes_tmp.flatten()
    # measure RMS of corrected lightcurve to check efficiency
    sig_after_gp, mad_after_gp = np.nanstd(ffluxes_tmp), np.nanmedian(np.abs(ffluxes_tmp-np.nanmedian(ffluxes_tmp))) / 0.67449
    
    # print RMS of data  before and after
    print("STATS Before WPPOS detrend: sigma={:.4f}% mad={:.4f}%".format(sig_before_gp*100, mad_before_gp*100))
    print("STATS After WPPOS detrend: sigma={:.4f}% mad={:.4f}%".format(sig_after_gp*100, mad_after_gp*100))

    return out_fluxes, out_fluxerrs


def diff_light_curve(times, fluxes, fluxerrs, posterior, nsig=100, model_time_sampling=0.001, offset_nsig=10, combine_by_median=False, binsize = 0.005, output="", instrum_index=0, use_calibs=True, plot_comps=False, plot=False, verbose=False):
    """
        Detrended differential light curve
    """
    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    
    planet_params = posterior["planet_params"]
    instrum_params = posterior["instrum_params"]
    instrument_indexes = posterior["instrument_indexes"]
    theta, labels = posterior["theta"], posterior["labels"]
    
    if verbose:
        # print out best fit parameters and errors
        print("----------------")
        print("Planet parameters:")
        for key in planet_params.keys() :
            if not key.endswith("_err") and not key.endswith("_pdf") :
                print(key, "=", planet_params[key])
        print("----------------")

    nstars = len(times)
    
    time = times[0]
    fluxes = np.array(fluxes)
    fluxerrs = np.array(fluxerrs)
    
    mintime, maxtime = time[0], time[-1]
        
    # Definir o array de tempos do modelo.
    t = np.arange(mintime, maxtime, model_time_sampling)
    transit_models = np.full_like(t, 1.0)
    obs_transit_models = np.full_like(time, 1.0)
    
    for j in range(int(posterior["n_planets"])) :
        planet_transit_id = "{0}_{1:03d}".format('transit', j)
        if planet_params[planet_transit_id] :
            transit_models *= exoplanetlib.batman_transit_model(t, planet_params, planet_index=j)
            obs_transit_models *= exoplanetlib.batman_transit_model(time, planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)
            
    if plot_comps :
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
        axs0 = axs[0]
    else :
        fig, axs0 = plt.subplots(1, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0})
            
    if plot :
        axs0.plot(t, (transit_models-1.0)*100, "g-", lw=3, zorder=2, label="Transit model")

    lcs = fluxes
    elcs = fluxerrs
    
    if  use_calibs :
        calibs = []
        for i in range(nstars):
            calib = exoplanetlib.calib_model(nstars, i, posterior['calib_params'], times[i])
            calibs.append(calib)
        calibs = np.array(calibs)
    
        lcs /= calibs
        elcs /= calibs

    target_median_fluxes = np.nanmedian(fluxes, axis=1)
    delta_mags = -2.5 * np.log10(target_median_fluxes)

    target_median_lcs = np.nanmedian(lcs/obs_transit_models, axis=1)
    
    for i in range(nstars):
        mlc =  np.nanmedian(lcs[i]/obs_transit_models)
        lcs[i] /= mlc
        elcs[i] /= mlc
        
    offset = 0

    lc_mean = np.average(lcs, axis=0, weights=1./(elcs*elcs))
    elc_mean = np.nanstd(lcs-lc_mean, axis=0)
    
    if combine_by_median :
        lc_mean = np.nanmedian(lcs, axis=0)
        elc_mean = np.nanmedian(np.abs(lcs-lc_mean), axis=0) / 0.67449

    #for i in range(len(lc_mean)) :
    #    lc_mean[i], elc_mean[i] = fitlib.odd_ratio_mean(lcs[:,i], elcs[:,i])

    mlc = np.nanmedian(lc_mean/obs_transit_models)
    rms = np.nanmedian(np.abs(lc_mean/obs_transit_models-mlc)) / 0.67449
    
    keep = elc_mean/obs_transit_models < nsig*rms
    keep &= np.abs(1. - lc_mean/obs_transit_models) < nsig*rms

    bin_time, bin_flux, bin_fluxerr = fitlib.bin_data(time[keep], lc_mean[keep], elc_mean[keep], median=False, binsize=binsize)

    if plot :
        bin_transit_models = np.full_like(bin_time, 1.0)
        for j in range(int(posterior["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                bin_transit_models *= exoplanetlib.batman_transit_model(bin_time, planet_params, planet_index=j, instrum_params=instrum_params, instrum_index=instrum_index)

        axs0.errorbar(time[keep], (lc_mean[keep]-1.0)*100, yerr=elc_mean[keep]*100, fmt='o', color="grey", alpha=0.15, label=r"Master: $\sigma$={:.2f}%".format(rms*100),zorder=1)
        
        bin_mlc = np.nanmedian(bin_flux/bin_transit_models)
        binrms = np.nanmedian(np.abs(bin_flux/bin_transit_models-bin_mlc)) / 0.67449
            
        axs0.errorbar(bin_time, (bin_flux-1.0)*100, yerr=bin_fluxerr*100, fmt='o', color="k", alpha=0.8, zorder=2, label=r"Master binned by {:.2f}h: $\sigma$={:.3f}%".format(binsize*24.0,binrms*100))
        
        axs0.set_ylabel(r"relative flux (%)", fontsize=20)
        #axs0.legend(fontsize=16)
        axs0.tick_params(axis='x', labelsize=14)
        axs0.tick_params(axis='y', labelsize=14)
        axs0.minorticks_on()
        axs0.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs0.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
    
        if plot_comps :
            offset = np.nanpercentile(lc_mean, 1.0) - offset_nsig*rms

            axs[1].hlines(0, mintime, maxtime, colors='k', linestyle='--', lw=2, zorder=2)
            #plt.hlines(-rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5,zorder=2)
            #plt.hlines(+rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5,zorder=2)

            for i in range(nstars):
                mlc = np.nanmedian(lcs[i]/obs_transit_models)
                rms = np.nanmedian(np.abs(lcs[i]/obs_transit_models-mlc)) / 0.67449

                keep = elcs[i]/obs_transit_models < nsig*rms
                keep &= np.abs(1. - lcs[i]/obs_transit_models) < nsig*rms

                comp_label = "C{:03d}".format(i)
                axs[1].errorbar(time[keep], (lcs[i][keep]/obs_transit_models[keep]-mlc)*100, yerr=(elcs[i][keep]/obs_transit_models[keep])*100, fmt='.', alpha=0.1, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} %".format(comp_label, delta_mags[i], rms*100),zorder=1)

            axs[1].tick_params(axis='x', labelsize=14)
            axs[1].tick_params(axis='y', labelsize=14)
            axs[1].minorticks_on()
            axs[1].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
            axs[1].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
            #axs[1].set_ylabel(r"fluxo (%)", fontsize=20)
            axs[1].set_xlabel("time (BJD)", fontsize=20)
        else :
            axs0.set_xlabel("time (BJD)", fontsize=20)
            
        #plt.xlabel(r"time (BJD)", fontsize=16)
        #plt.ylabel(r"$\Delta$mag", fontsize=16)
        #plt.xlabel(r"time (BTJD)", fontsize=16)
        #plt.ylabel(r"Flux ratio", fontsize=16)
        #plt.legend(loc='lower left',fontsize=10)
        plt.show()

    tbl = Table()
    tbl["TIME"] = time
    tbl["TRANSIT_MODEL"] = obs_transit_models
    tbl["FLUX"] = lc_mean
    tbl["FLUXERR"] = elc_mean

    if output != "" :
        header = fits.Header()
        
        for key in planet_params.keys() :
            if not key.endswith("_err") and not key.endswith("_pdf") :
                header.set(key, planet_params[key])

        primary_hdu = fits.PrimaryHDU(header=header)
        hdu_time = fits.ImageHDU(data=time, name="TIME")
        hdu_mag = fits.ImageHDU(data=lc_mean, name="FLUX")
        hdu_magerr = fits.ImageHDU(data=elc_mean, name="FLUXERR")
        hdu_bintime = fits.ImageHDU(data=bin_time, name="BIN_TIME")
        hdu_binmag = fits.ImageHDU(data=bin_flux, name="BIN_FLUX")
        hdu_binmagerr = fits.ImageHDU(data=bin_fluxerr, name="BIN_FLUXERR")

        listofhuds = [primary_hdu, hdu_time, hdu_mag, hdu_magerr, hdu_bintime, hdu_binmag, hdu_binmagerr]
        mef_hdu = fits.HDUList(listofhuds)
        mef_hdu.writeto(options.output_lc, overwrite=True)

    return tbl



def color_light_curve(time, model1, flux1, fluxerr1, model2, flux2, fluxerr2, label1="g-band",label2="z-band", title="", plot=False) :

    yerr = np.full_like(time, np.nan)
    y = flux1 / flux2
    ym = model1 / model2
    for i in range(len(time)) :
        y1 = ufloat(flux1[i],fluxerr1[i])
        y2 = ufloat(flux2[i],fluxerr2[i])
        yval = y1/y2
        yerr[i] = yval.std_dev

    if plot :
        if title != "" :
            plt.title(title)
        plt.plot(time, ym, '-',color="darkgreen",lw=3)
        plt.plot(time, y,'-',color="#ff7f0e")
        plt.fill_between(time, y+yerr, y-yerr, color="#ff7f0e", alpha=0.3, edgecolor="none")
        plt.xlabel("Time (BJD)",fontsize=20)
        plt.ylabel(r"Flux ratio ({} / {})".format(label1,label2),fontsize=20)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.minorticks_on()
        plt.tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        plt.tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        plt.show()

    return ym, y, yerr



def plot_s4_transit_light_curves(s4times, s4fluxes, s4fluxerrs, s4posteriors, model_time_sampling=0.001, object_name="") :


    """ Module to plot differential light curves and auxiliary data
    
    Parameters
    ----------
    s4times: list of 4 np.array()
        list of 4 time arrays
    s4fluxes: list of 4 np.array()
        list of 4 flux arrays
    s4fluxerrs: list of 4 np.array()
        list of 4 flux error arrays
    
    Returns

    -------
    """

    ndatasets = len(s4times)

    bands = ['g', 'r', 'i', 'z']
    colors = ['darkblue','darkgreen','darkorange','darkred']
        
    fig, axs = plt.subplots(ndatasets, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

    for ch in range(ndatasets) :
               
        planet_params = s4posteriors[ch]["planet_params"]
        mintime, maxtime = s4times[ch][0], s4times[ch][-1]
        # Definir o array de tempos do modelo.
        t = np.arange(mintime, maxtime, model_time_sampling)
        transit_models = np.full_like(t, 1.0)
        obs_transit_models = np.full_like(s4times[ch], 1.0)

        for j in range(int(s4posteriors[ch]["n_planets"])) :
            planet_transit_id = "{0}_{1:03d}".format('transit', j)
            if planet_params[planet_transit_id] :
                transit_models *= exoplanetlib.batman_transit_model(t, planet_params, planet_index=j)
                obs_transit_models *= exoplanetlib.batman_transit_model(s4times[ch], planet_params, planet_index=j)
                                
        if ch == 0 :
            axs[ch].set_title(r"{}".format(object_name), fontsize=20)
    
        #axs[ch].plot(t, (transit_models-1.0)*100, "g-", lw=3, zorder=2, label="Transit model")
        #axs[ch].plot(t, transit_models, "k-", lw=3, label="Transit model", zorder=2)
        axs[ch].plot(t, transit_models, "k-", lw=3, zorder=3)

        residuals = s4fluxes[ch] - obs_transit_models
        
        sampling = np.median(np.abs(s4times[ch][1:] - s4times[ch][:-1])) * 24 * 60 * 60

        print("RMS of residuals in the {}-band (CHANNEL {}): {:.6f}; Time sampling: {:.1f} s".format(bands[ch],ch+1,np.nanstd(residuals),sampling))

        #x, y, yerr = bin_data(lc['TIME'], lc['diffmagsum'], lc['magsum_err'], median=False, binsize = binsize)
    
        axs[ch].errorbar(s4times[ch], s4fluxes[ch], yerr=s4fluxerrs[ch], color=colors[ch],fmt="o", alpha=0.3, label="{}-band".format(bands[ch]), zorder=2)
        #axs[ch].plot(lc['TIME'],lc['diffmagsum'],".",color=colors[ch], alpha=0.3, label=bands[ch])

        axs[ch].tick_params(axis='x', labelsize=14)
        axs[ch].tick_params(axis='y', labelsize=14)
        axs[ch].minorticks_on()
        axs[ch].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[ch].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
        if ch == 3 :
            axs[ch].set_xlabel(r"time (BJD)", fontsize=16)
                
        axs[ch].set_ylabel(r"flux", fontsize=16)
        axs[ch].legend(fontsize=16)

    plt.show()
        
    return 0
