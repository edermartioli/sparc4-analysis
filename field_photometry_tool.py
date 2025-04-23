"""
    Created on Sep 13 2024
    
    Description: A tool to analyze field photometry with SPARC4
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:

    python field_photometry_tool.py --object="Eta Car" --input="/Users/eder/Science/Transits-OPD_2024A/20240619" --obj_indexes="0,0,0,0"
    
    python field_photometry_tool.py --simbad_id="PSR B0531+21" --object="M1" --input="/Users/eder/Science/Crab/phot/"
    
    python field_photometry_tool.py --simbad_id="NGC4045" --object="NGC4045" --input="/Users/eder/Science/sparc4-science/NGC4045"
    python field_photometry_tool.py --simbad_id="NGC7089" --object="NGC7089" --input="/Users/eder/Science/NGC7089"
        
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

exo_transit_tool_path = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data directory',type='string', default="./")
parser.add_option("-s", "--simbad_id", dest="simbad_id", help='Object Simbad ID',type='string', default="")
parser.add_option("-j", "--object", dest="object", help='Object ID in header',type='string', default="")
parser.add_option("-x", "--obj_indexes", dest="obj_indexes", help='Object indexes',type='string', default="")
parser.add_option("-o", "--output", dest="output", help='Output master catalog',type='string', default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with field_photometry_tool.py -h "); sys.exit(1);

calibrate_photometry = True
calibrate_astrometry = True

# input data directory
datadir = options.input

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


# find Simbad match for main object
object_match_simbad, object_coords = s4analyb.match_object_with_simbad(options.simbad_id, search_radius_arcsec=10)

# - recalibrate astrometry -- not needed if product has a good enough astrometry
if calibrate_astrometry :
    use_catalog = False
    if objprods["instmode"] == "POLAR" :
        use_catalog = True
    catalogs, objprods = s4analyb.calibrate_astrometry_in_stack_catalog(objprods, use_catalog=use_catalog, aperture_radius=10, niter=3, plot=True)

exit()

# project stack images of all channels onto the same WCS frame as the selected base channel, and produce color match products.
objprods = s4analyb.color_match_products(objprods, convolve_images=False, kernel_size=2.0, base_ch=0, reproject_order='bicubic')

object_indexes = []
if options.obj_indexes == "" :
    #  identify main source in the observed catalogs
    object_indexes = s4analyb.get_object_indexes_in_catalogs(objprods, obj_id=options.object, simbad_id=options.simbad_id)
else :
    objidx = options.obj_indexes.split(",")
    for i in range(len(objidx)) :
        object_indexes.append(int(objidx[i]))
print(object_indexes)

#  calibrate photometry to a standard system
if calibrate_photometry :
    #catalogs, objprods = s4analyb.calibrate_photometry_in_stack_catalog(objprods, aperture_radius=10, vizier_catalog_name='SkyMapper', plot=True)
    catalogs, objprods = s4analyb.calibrate_photometry_in_stack_catalog(objprods, aperture_radius=10, vizier_catalog_name='GAIADR2', plot=True)
    for ch in range(4) :
        print("Band {}: (a,b)={}".format(ch+1,objprods["phot_calibration"][ch]))

master_catalog = s4analyb.create_master_catalog()
try :
    # create a master catalog matching sources for all 4 channels
    master_catalog = s4analyb.master_catalog_in_four_bands(objprods, calibrate_photometry=calibrate_photometry)
except :
    print("Could not populate master catalog")
    
s4analyplt.plot_s4data_for_diffphot(objprods, object_indexes, comps=[[],[],[],[]], object_name=options.object)

if options.output != "" :
    master_catalog.write(options.output, overwrite=True)

