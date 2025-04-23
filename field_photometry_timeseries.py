"""
    Created on Sep 13 2024
    
    Description: A tool to analyze field photometry with a SPARC4 time series
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:

    python field_photometry_timeseries.py --input="/Users/eder/Science/Transits-OPD_2024A/HATS-24/" --output="/Users/eder/Science/Transits-OPD_2024A/HATS-24/hats-24_master-catalog.csv"
            
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

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data directory',type='string', default="./")
parser.add_option("-o", "--output", dest="output", help='Output master catalog',type='string', default="")
parser.add_option("-c", "--calib_catalog", dest="calib_catalog", help='Calibrated catalog: SkyMapper or GAIADR2',type='string', default="SkyMapper")
parser.add_option("-a", "--aperture_radius", dest="aperture_radius", help='Aperture radius in pixels',type='int', default=12)
parser.add_option("-b", "--pol_beam", dest="pol_beam", help='Polarimetric beam: N or S',type='string', default='N')
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with field_photometry_timeseries.py -h "); sys.exit(1);

pol_beam = options.pol_beam  # 'N' or 'S'

calibrate_photometry = True
calibrate_astrometry = True

# input data directory
datadir = options.input

# load SPARC4 data products in the directory
s4products = s4analyb.load_data_products(datadir, pol_beam=pol_beam)

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
    use_catalog = False
    if objprods["instmode"] == "POLAR" :
        use_catalog = True
    catalogs, objprods = s4analyb.calibrate_astrometry_in_stack_catalog(objprods, use_catalog=use_catalog, aperture_radius=options.aperture_radius, niter=3, sip_degree=7, twirl_find_peak_threshold=1, fov_search_factor=1.3, use_twirl_to_compute_wcs=False, plot=options.plot, pol_beam=pol_beam)

# project stack images of all channels onto the same WCS frame as the selected base channel, and produce color match products.
objprods = s4analyb.color_match_products(objprods, convolve_images=False, kernel_size=2.0, base_ch=0, reproject_order='bicubic')

if calibrate_photometry :
    #  calibrate photometry to a standard system
    catalogs, objprods = s4analyb.calibrate_photometry_in_stack_catalog(objprods, aperture_radius=options.aperture_radius, vizier_catalog_name= options.calib_catalog, dmag_sig_clip=0.3, plot=True, pol_beam=pol_beam)

# initialize master catalog, combining all four channels
master_catalog = s4analyb.create_master_catalog()
try :
    # create a master catalog matching sources for all 4 channels
    master_catalog = s4analyb.master_catalog_in_four_bands(objprods, calibrate_photometry=calibrate_photometry, pol_beam=pol_beam)
except :
    print("Could not populate master catalog")

# run photometric solution in time series
master_phot_timeseries_catalog = s4analyb.photometric_solution_in_timeseries(objprods, master_catalog, aperture_radius=options.aperture_radius, use_median=True, time_binsize=0.01, plot=True, pol_beam=pol_beam)

# print out catalog
if options.verbose:
    print(master_phot_timeseries_catalog)

# save master catalog to output file
if options.output != "" :
    master_phot_timeseries_catalog.write(options.output, overwrite=True)


