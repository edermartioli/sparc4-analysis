"""
    Created on Aug 23 2024

    Description: Library of recipes for plotting analysis data of SPARC4 data

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import astropy.io.fits as fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import CircleSkyRegion

import photutils

def plot_s4data_for_diffphot(objprods, object_indexes, comps, percentile=98, aperture_radius=10, object_name="") :


    """ Module to plot differential light curves and auxiliary data
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    object_indexes: list: [int, int, int, int]
        list of target indexes
    comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
        list of arrays of comparison indexes in the four channels
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    
    Returns
        
    -------
    """

    lcs = []
    nstars = 1

    bands = ['g', 'r', 'i', 'z']
    colors = ['lightblue','darkgreen','darkorange','red']
    panel_pos = [[0,0],[0,1],[1,0],[1,1]]
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    if objprods["instmode"] == "POLAR" :
        catalog_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, layout='constrained', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(8, 8))
        
    for ch in range(4) :
        col, row = panel_pos[ch]
        
        axs[col,row].set_title(r"{}-band".format(bands[ch]))

        if objprods["stack"][ch] is not None :
            
            hdul = fits.open(objprods["stack"][ch])
            img_data, hdr = hdul[0].data, hdul[0].header
            catalog = Table(hdul[catalog_ext].data)
            wcs = WCS(hdr, naxis=2)
        
            for j in range(len(catalog["X"])) :
                
                #axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                
                if j == object_indexes[ch] :
                    axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j]-1.1*aperture_radius, '{}'.format(j), c='gray', fontsize=8)
                    
                    axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(object_name), c='white', fontsize=12, alpha=0.5)
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                else :
                    if j in comps[ch] :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], 'C{}'.format(j), c='white', fontsize=12)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    else :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(j), c='gray', fontsize=8)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor="grey"))
                            
        else :
            img_data = np.empty([1024, 1024])
  
        axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')
                  
        axs[col,row].tick_params(axis='x', labelsize=10)
        axs[col,row].tick_params(axis='y', labelsize=10)
        axs[col,row].minorticks_on()
        axs[col,row].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[col,row].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
                         
    plt.show()
        
    return



def plot_s4data_for_polarimetry(objprods, object_indexes, percentile=98, aperture_radius=10, object_name="") :


    """ Module to plot differential light curves and auxiliary data
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    object_indexes: list: [int, int, int, int]
        list of target indexes
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    
    Returns
        
    -------
    """

    lcs = []
    nstars = 1

    bands = ['g', 'r', 'i', 'z']
    colors = ['lightblue','darkgreen','darkorange','red']
    panel_pos = [[0,0],[0,1],[1,0],[1,1]]
    
    
    catalog_n_ext = "CATALOG_POL_N_AP{:03d}".format(aperture_radius)
    catalog_s_ext = "CATALOG_POL_S_AP{:03d}".format(aperture_radius)
    
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, layout='constrained', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(8, 8))
        
    for ch in range(4) :
        col, row = panel_pos[ch]
        
        axs[col,row].set_title(r"{}-band".format(bands[ch]))

        if objprods["stack"][ch] is not None :
            
            hdul = fits.open(objprods["stack"][ch])
            img_data, hdr = hdul[0].data, hdul[0].header
            catalog = Table(hdul[catalog_n_ext].data)
            wcs = WCS(hdr, naxis=2)
                
            x_o, y_o = hdul[catalog_n_ext].data['x'], hdul[catalog_n_ext].data['y']
            x_e, y_e = hdul[catalog_s_ext].data['x'], hdul[catalog_s_ext].data['y']

            mean_aper = np.mean(hdul[catalog_n_ext].data['APER'])

            #plt.figure(figsize=(10, 10))

            axs[col,row].plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
            axs[col,row].plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

            for i in range(len(x_o)):
                x = [x_o[i], x_e[i]]
                y = [y_o[i], y_e[i]]
                axs[col,row].plot(x, y, 'w-o', alpha=0.5)
                if i == object_indexes[ch] :
                    axs[col,row].annotate("{}".format(i), [np.mean(x)-25, np.mean(y)+25], color='gray')
                    axs[col,row].annotate('{}'.format(object_name), [np.mean(x)-25, np.mean(y)+25], color='w',fontsize=12, alpha=0.5)
                else :
                    axs[col,row].annotate("{}".format(i), [np.mean(x)-25, np.mean(y)+25], color='w')
            """
            for j in range(len(catalog["X"])) :
                
                #axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                
                if j == object_indexes[ch] :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j]-1.1*aperture_radius, '{}'.format(j), c='gray', fontsize=8)
                
                    axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(object_name), c='white', fontsize=12, alpha=0.5)
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius+2, facecolor="none", edgecolor=colors[ch]))
                else :
                    if j in comps[ch] :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], 'C{}'.format(j), c='white', fontsize=12)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor=colors[ch]))
                    else :
                        axs[col,row].text(catalog["X"][j]+1.1*aperture_radius, catalog["Y"][j], '{}'.format(j), c='gray', fontsize=8)
                        axs[col,row].add_patch(plt.Circle((catalog["X"][j],catalog["Y"][j]), radius=aperture_radius, facecolor="none", edgecolor="grey"))
            """
        else :
            img_data = np.empty([1024, 1024])
  
        axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100-percentile),vmax=np.percentile(img_data, percentile), origin='lower', aspect='equal')
                  
        #axs[col,row].imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')
                
        axs[col,row].tick_params(axis='x', labelsize=10)
        axs[col,row].tick_params(axis='y', labelsize=10)
        axs[col,row].minorticks_on()
        axs[col,row].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
        axs[col,row].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
                         
    plt.show()
        
    return
