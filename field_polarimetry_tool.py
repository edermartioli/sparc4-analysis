"""
    Created on Sep 9 2024
    
    Description: A tool to analyze field polarimetry with SPARC4
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.

    Simple usage examples:
    
    python field_polarimetry_tool.py --object="NGC2024" --input="/Users/eder/Science/sparc4-science/ngc2024" --obj_indexes="0,0,0,0"
    
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
parser.add_option("-j", "--object", dest="object", help='Object ID',type='string', default="")
parser.add_option("-x", "--obj_indexes", dest="obj_indexes", help='Object indexes',type='string', default="")
parser.add_option("-o", "--output", dest="output", help='Output master catalog',type='string', default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with field_polarimetry_tool.py -h "); sys.exit(1);

calibrate_photometry = False
calibrate_astrometry = True
create_master_catalog = False

# input data directory
datadir = options.input

# load SPARC4 data products in the directory
s4products = s4analyb.load_data_products(datadir, object_id=options.object)

objprods={}
for key in s4products.keys() :
    if 'POLAR' in key :
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

aperture_radius=12

# find Simbad match for main object
object_match_simbad, object_coords = s4analyb.match_object_with_simbad(options.object, search_radius_arcsec=10)
    
# - recalibrate astrometry -- not needed if product has a good enough astrometry
if calibrate_astrometry :
    catalogs, objprods = s4analyb.calibrate_astrometry_in_stack_catalog(objprods, use_catalog=True, aperture_radius=12, niter=3, plot=True)
    
object_indexes = []
if options.obj_indexes == "" :
    #  identify main source in the observed catalogs
    object_indexes = s4analyb.get_object_indexes_in_catalogs(objprods, obj_id=options.object, simbad_id=options.object)
else :
    objidx = options.obj_indexes.split(",")
    for i in range(len(objidx)) :
        object_indexes.append(int(objidx[i]))
print(object_indexes)

#  calibrate photometry to a standard system
if calibrate_photometry :
    catalogs, objprods = s4analyb.calibrate_photometry_in_stack_catalog(objprods, aperture_radius=10, vizier_catalog_name='SkyMapper', plot=True)

if calibrate_photometry :
    for ch in range(4) :
        print("Band {}: (a,b)={}".format(ch+1,objprods["phot_calibration"][ch]))
            
if create_master_catalog :
    master_catalog = s4analyb.create_master_catalog()
    try :
        # create a master catalog matching sources for all 4 channels
        master_catalog = s4analyb.master_catalog_in_four_bands(objprods, calibrate_photometry=calibrate_photometry)
    except :
        print("Could not populate master catalog")
    
s4analyplt.plot_s4data_for_polarimetry(objprods, object_indexes, object_name=options.object)


catalog_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)

for ch in range(4) :

    if objprods["polarl2"][ch] is not None :
        print(objprods["polarl2"][ch])
        pol_results = s4pipelib.get_polarimetry_results(objprods["polarl2"][ch], source_index=object_indexes[ch], min_aperture=0, max_aperture=1024, compute_k=True, plot=True, verbose=True)
                
    if objprods["polarl4"][ch] is not None :
        print(objprods["polarl4"][ch])
        pol_results = s4pipelib.get_polarimetry_results(objprods["polarl4"][ch], source_index=object_indexes[ch], min_aperture=0, max_aperture=1024, compute_k=True, plot=True, verbose=True)
        
    if objprods["stack"][ch] is not None and objprods["polarl2"][ch] is not None :
        s4plt.plot_polarimetry_map(objprods["stack"][ch], objprods["polarl2"][ch], min_aperture=0, max_aperture=1024, percentile=99.5,ref_catalog=catalog_ext, src_label_offset=30, arrow_size_scale=None, title_label="{}".format(options.object))

if options.output != "" and create_master_catalog:
    master_catalog.write(options.output, overwrite=True)
