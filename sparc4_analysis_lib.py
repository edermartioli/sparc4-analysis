"""
    Created on Aug 21 2024

    Description: Library of recipes for the analysis of SPARC4 data

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """


from astroquery.simbad import Simbad

from astropy import units as u
from astroplan import Observer
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table, join, Column
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.nddata import NDData
from reproject import reproject_interp

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

from astropop.catalogs import gaia
from uncertainties import ufloat,umath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import photutils

from scipy.optimize import curve_fit
from scipy import odr

import sparc4.product_plots as s4plt
import sparc4.pipeline_lib as s4pipelib
import sparc4.products as s4p

from copy import deepcopy

import glob


def load_data_products(datadir, object_id=None, pol_beam='S+N') :
    """ Module to identify and store the paths of s4 data products within a given directory
    Parameters
    ----------
    datadir : path
        directory path to search products
    obj_id : str
        object identification name
    Returns
        loc : dict
            dictionary data container to store product paths.
            The keys for each entry follow the structure objectname_instmode
    -------
    """

    loc = {}

    # get list of stack products
    stackfiles = sorted(glob.glob("{}/*stack.fits".format(datadir)))
    
    # get instrument mode (PHOT or POLAR) from stack images that match object id
    objs_istmode = []
    for i in range(len(stackfiles)) :
        hdr = fits.getheader(stackfiles[i])
        if hdr["OBJECT"] not in objs_istmode and (object_id is None or object_id.replace(" ","").upper() == hdr["OBJECT"].replace(" ","").upper()) :
            objs_istmode.append([hdr["OBJECT"],hdr["INSTMODE"]])

    # for each object / instrument mode find all other products
    for obj,instmode in objs_istmode :
        key = "{}_{}".format(obj,instmode)
    
        loc[key] = {}
        
        loc[key]["object"] = obj
        loc[key]["instmode"] = instmode

        loc[key]["stack"] = [None,None,None,None]
        loc[key]["ts"] = [None,None,None,None]
        loc[key]["lc"] = [None,None,None,None]
        loc[key]["polarl2"] = [None,None,None,None]
        loc[key]["polarl4"] = [None,None,None,None]

        for i in range(len(stackfiles)) :
            hdr = fits.getheader(stackfiles[i])
            if loc[key]["stack"][hdr["CHANNEL"]-1] is None :
                loc[key]["stack"][hdr["CHANNEL"]-1] = stackfiles[i]
            
        if instmode == "POLAR":
        
            lcfiles = sorted(glob.glob("{}/*{}_lc.fits".format(datadir,pol_beam)))
           
            tsfiles = sorted(glob.glob("{}/*ts.fits".format(datadir)))
            polarl2files = sorted(glob.glob("{}/*_POLAR_L2*polar.fits".format(datadir)))
            polarl4files = sorted(glob.glob("{}/*_POLAR_L4*polar.fits".format(datadir)))

            for i in range(len(tsfiles)) :
                hdr = fits.getheader(tsfiles[i])
                if loc[key]["ts"][hdr["CHANNEL"]-1] is None :
                    loc[key]["ts"][hdr["CHANNEL"]-1] = tsfiles[i]
                    
            for i in range(len(polarl2files)) :
                hdr = fits.getheader(polarl2files[i])
                if loc[key]["polarl2"][hdr["CHANNEL"]-1] is None :
                    loc[key]["polarl2"][hdr["CHANNEL"]-1] = polarl2files[i]
                    
            for i in range(len(polarl4files)) :
                hdr = fits.getheader(polarl4files[i])
                if loc[key]["polarl4"][hdr["CHANNEL"]-1] is None :
                    loc[key]["polarl4"][hdr["CHANNEL"]-1] = polarl4files[i]
        else :
            lcfiles = sorted(glob.glob("{}/*lc.fits".format(datadir)))
        
        for i in range(len(lcfiles)) :
            hdr = fits.getheader(lcfiles[i])
            
            parts = lcfiles[i].split("_")
            
            for part in parts :
                if 's4c' in part :
                    ch = int(part[-1])
                    break
                    
            if "CHANNEL" in hdr.keys() :
                ch = hdr["CHANNEL"]
        
            if loc[key]["lc"][ch-1] is None :
                loc[key]["lc"][ch-1] = lcfiles[i]
                
    return loc



def match_catalog_with_simbad(obj_id, s4_proc_frame, catalog_ext="CATALOG_POL_N_AP010", tolerance=2., verbose=False) :

    """ Module to match a catalog from sci image product (proc.fits) with Simbad
    Parameters
    ----------
    obj_id : str
        object identification name
    s4_proc_frame : str
        sparc4 image fits product (proc.fits) file path
    catalog_ext : str or int (optional)
        FITS hdu extension name or number for catalog
    tolerance : float (optional)
        maximum tolerance for angular distance to match object [arcsec]
    verbose : bool (optional)
        to print messages
    Returns
        imin, coord : tuple : int, SkyCoord
        imin is the object matched index in the catalog and coord is its respective coodinates
    -------
    """

    # query SIMBAD repository to match object by name
    obj_match_simbad = Simbad.query_object(obj_id)
        
    # cast coordinates into SkyCoord
    coord = SkyCoord(obj_match_simbad["RA"][0], obj_match_simbad["DEC"][0], unit=(u.hourangle, u.deg), frame='icrs')
    
    # open sparc4 image product as hdulist
    hdul = fits.open(s4_proc_frame)
    
    # read catalog data as Table
    catalog = Table(hdul[catalog_ext].data)

    # search catalog objects with minimum distance from simbad object
    delta_min, imin  = tolerance, -1
    for i in range(len(catalog)) :
        cat_ra, cat_dec = catalog['RA'][i], catalog['DEC'][i]
        
        dra = np.cos(coord.dec.deg*np.pi/180.)*(cat_ra - coord.ra.deg)*60*60
        ddec = (cat_dec - coord.dec.deg)*60*60
        delta = np.sqrt((dra)**2 + (ddec)**2)
        
        if delta < delta_min :
            delta_min = delta
            imin = i
   
    if imin == -1 :
        print("Object {} did not match a Simbad object".format(obj_id))
        return None, None
   
    if verbose :
        ra_diff = (catalog[imin][1]-coord.ra.deg)*60*60
        dec_diff = (catalog[imin][2]-coord.dec.deg)*60*60

        print("Object {} index is {} with RA={:.5f} (diff={:.2f} arcsec) / DEC={:.5f} (diff={:.2f} arcsec)".format(obj_id, imin, catalog[imin][1], ra_diff, catalog[imin][2], dec_diff))

    return imin, coord
    
    
def get_object_indexes_in_catalogs(objprods, catalog_ext=1, simbad_id=None, verbose=False) :

    """ Module to get object indexes in catalogs of s4 image products
    Parameters
    ----------
    objprods: dict
        data container to store products
    catalog_ext: int or str
        FITS extension for catalog
    simbad_id: str
        simbad identification name
    verbose: bool
        print verbose messages
    Returns
        object_indexes : list : [int, int, int, int]
        list of detected object indexes in the four channels
    -------
    """

    object_indexes = [None, None, None, None]
    
    if simbad_id is None :
        simbad_id = objprods["object"]
        
    for ch in range(4) :
        if objprods["stack"][ch] :
            imin, coord = match_catalog_with_simbad(simbad_id, objprods["stack"][ch], catalog_ext=catalog_ext, verbose=verbose)
            if imin is not None :
                object_indexes[ch] = imin
                
    return object_indexes
    
    
def match_object_with_simbad(obj_id, ra=None, dec=None, search_radius_arcsec=10) :

    """ Module to match a given object id with Simbad
    Parameters
    ----------
    objprods: dict
        data container to store products
    ra: float (optional)
        right ascension (deg)
    dec: float (optional)
        declination (deg)
    search_radius_arcsec: float
        search radius in units of arcseconds
    Returns
        obj_match_simbad, coord: simbad_entry, SkyCoord()
    -------
    """

    obj_match_simbad, coord = None, None
    
    try :
        print("Querying SIMBAD database to match object ID={}".format(obj_id))
        # query SIMBAD repository to match object by ID
        obj_match_simbad = Simbad.query_object(obj_id)
    except :
        if ra is not None and dec is not None :
            print("Querying SIMBAD database to match an object at RA={} DEC={}".format(ra, dec))
            # cast input coordinates into SkyCoord
            coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
            # query SIMBAD repository to match an object by coordinates
            obj_match_simbad = Simbad.query_region(coord, radius = search_radius_arcsec * (1./3600.) * u.deg)
        else :
            print("WARNING: could not find Simbad match for object {}".format(obj_id))

    if obj_match_simbad is not None :
        ra = obj_match_simbad["RA"][0]
        dec = obj_match_simbad["DEC"][0]
    
        # cast coordinates into SkyCoord
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

    return obj_match_simbad, coord
    

def get_sci_image_catalog(s4_proc_frame, catalog_ext="CATALOG_POL_N_AP010", wcs=None, print_catalog=False) :

    """ Module to get catalog from s4 image product and update ra and dec coords using an input wcs
    Parameters
    ----------
    s4_proc_frame : str
        sparc4 image fits product (proc.fits) file path
    catalog_ext : str or int (optional)
        FITS hdu extension name or number for catalog
    tolerance : float (optional)
        maximum tolerance for angular distance to match object [arcsec]
    wcs : astropy.wcs.WCS
        world coordinate system object used to update coords in output catalog
    print_catalog : bool
        to print catalog
    Returns
        catalog: astropy.table.Table
            updated catalog of sources
    -------
    """

    hdul = fits.open(s4_proc_frame)
    catalog = Table(hdul[catalog_ext].data)
    
    if wcs is not None :
        pixel_coords = np.ndarray((len(catalog["X"]), 2))
        for j in range(len(catalog["X"])) :
            pixel_coords[j] = [catalog["X"][j],catalog["Y"][j]]
        
        sky_coords = np.array(wcs.pixel_to_world_values(pixel_coords))

        for j in range(len(catalog["RA"])) :
            catalog["RA"][j] = sky_coords[j][0]
            catalog["DEC"][j] = sky_coords[j][1]
            
    if print_catalog :
        print(catalog)
        
    return catalog
    
    
def match_catalog_sources_with_gaiadr3(s4_proc_frame, wcs=None, catalog_ext="CATALOG_POL_N_AP010", search_radius=5, max_G_mag=18, clean_unmatched=False, verbose=False, plot=False) :

    """ Module to match catalog sources with gaia DR3 sources
    Parameters
    ----------
    s4_proc_frame : str
        sparc4 image fits product (proc.fits) file path
    wcs : astropy.wcs.WCS
        world coordinate system object used to update coords in output catalog
    catalog_ext : str or int (optional)
        FITS hdu extension name or number for catalog
    search_radius : float (optional)
        search radius around center of the field [arcmin]
    max_G_mag : float (optional)
        maximum G magnitude to consider in the matched catalog
    clean_unmatched : float (optional)
        remove catalog sources not matched with Gaid DR3
    verbose : bool
        to print messages
    plot : bool
        to plot results
    Returns
        catalog: astropy.table.Table
            updated catalog of sources
    -------
    """

    # get image header
    hdr = fits.getheader(s4_proc_frame,0)
    
    # get wcs
    if wcs is None :
        wcs = WCS(hdr, naxis=2)
        # get catalog
        catalog = get_sci_image_catalog(s4_proc_frame, catalog_ext=catalog_ext, print_catalog=False)
    else :
        # get catalog
        catalog = get_sci_image_catalog(s4_proc_frame, catalog_ext=catalog_ext, print_catalog=False, wcs=wcs)
    
    #catalog.write("catalog.csv",overwrite=True)
    
    # define observation time
    obstime = Time(hdr['BJD'], format='jd', scale='tcb')

    # grab sources from catalog
    xs, ys = catalog['X'], catalog['Y']
    ras, decs = catalog['RA'], catalog['DEC']
    mags, emags  = catalog['MAG'], catalog['EMAG']

    # get the center of the image
    ra_cen, dec_cen = wcs.wcs.crval[0], wcs.wcs.crval[1]
    
    # set image center coordinates
    center = SkyCoord(ra_cen, dec_cen, unit=(u.deg,u.deg), frame='icrs')

    # define search radius in arcmin
    r = search_radius * u.arcminute
    
    # retrieve all gaia sources within the field of view
    gaia_sources = gaia.gaiadr3(center, radius=r)
    
    # filter sources brighter than max_G_mag
    gaia_sources = gaia_sources[gaia_sources.magnitude('G').nominal <= max_G_mag]
    
    # grab RA, DEC of gaia sources
    RA_gaiadr3, DEC_gaiadr3 = gaia_sources['ra'], gaia_sources['dec']

    # transform RA, Dec to pixel coordinates
    x_gaiadr3,y_gaiadr3 = wcs.all_world2pix(RA_gaiadr3,DEC_gaiadr3,1)

    # match gaia sources with catalog sources
    matched = gaia_sources.match_objects(ras, decs, limit_angle='1 arcsec',obstime=obstime)

    matched_tbl = matched.table()
    
    if verbose :
        n_gaia_matched_sources = len(matched_tbl)
    
        print("Number of observed sources in catalog: {}".format(len(ras)))
        print("Number of Gaia sources in the field: {}".format(len(gaia_sources)))
        print("Number of matched sources with Gaia: {}".format(n_gaia_matched_sources))

    catalog.rename_column('RA', 'OBS_RA')
    catalog.rename_column('DEC', 'OBS_DEC')
    for colname in matched_tbl.colnames :
        if colname == 'id' :
            catalog.add_column(matched_tbl[colname], index=0)
        else :
            catalog.add_column(matched_tbl[colname])
    catalog.rename_column('id', 'Gaia ID')
    catalog.add_column(matched_tbl['id'], index=0)

    unmatched = catalog['Gaia ID'] == ""

    if clean_unmatched :
        catalog = catalog[~unmatched]
        
    if plot :
        img_data = fits.getdata(s4_proc_frame,0)
        clean_catalog = catalog[~unmatched]
                
        gaia_sources_skycoords = []
        cat_sources_skycoords = []

        for i in range(len(catalog)) :
            cat_ra, cat_dec = catalog['OBS_RA'][i], catalog['OBS_DEC'][i]
            cat_sources_skycoords.append([cat_ra, cat_dec])
        cat_sources_skycoords = np.array(cat_sources_skycoords,dtype=float)

        for i in range(len(clean_catalog)) :
            gaia_ra, gaia_dec = clean_catalog['ra'][i], clean_catalog[i]['dec']
            gaia_sources_skycoords.append([gaia_ra, gaia_dec])
        gaia_sources_skycoords = np.array(gaia_sources_skycoords,dtype=float)
    
        gaia_sources_pixcoords = np.array(wcs.world_to_pixel_values(gaia_sources_skycoords))
        cat_sources_pixcoords = np.array(wcs.world_to_pixel_values(cat_sources_skycoords))
        plt.imshow(img_data, vmin=np.median(img_data), vmax=2 * np.median(img_data), cmap="Greys_r")
        
        _ = photutils.aperture.CircularAperture(gaia_sources_pixcoords, r=8.0).plot(color="y")
        _ = photutils.aperture.CircularAperture(cat_sources_pixcoords, r=12.0).plot(color="g")
        plt.show()
    
    return catalog
    
    
def match_catalog_sources_with_vizier(s4_proc_frame, wcs=None, catalog_ext="CATALOG_POL_N_AP010", search_radius=5, vizier_catalog_name='SkyMapper', max_mag=21, max_sources=1e4, match_threshold=2., plot=False) :

    """ Module to match catalog sources with sources of a given Vizier catalog
    Parameters
    ----------
    s4_proc_frame : str
        sparc4 image fits product (proc.fits) file path
    wcs : astropy.wcs.WCS
        world coordinate system object used to update coords in output catalog
    catalog_ext : str or int (optional)
        FITS hdu extension name or number for catalog
    search_radius : float (optional)
        search radius around center of the field [arcmin]
    vizier_catalog_name : str (optional)
        Vizier catalog name
    max_mag : float (optional)
        maximum magnitude to consider in the matched catalog
    max_sources : int (optional)
        maximum number of sources to truncate search
    match_threshold : float (optional)
        tolerance in angular distance to consider a match [arcsec]
    plot : bool
        to plot results
    Returns
        catalog: astropy.table.Table
            updated catalog of sources
    -------
    """

    # get image header
    hdr = fits.getheader(s4_proc_frame,0)
    
    # get wcs
    if wcs is None :
        wcs = WCS(hdr, naxis=2)
        # get catalog
        catalog = get_sci_image_catalog(s4_proc_frame, catalog_ext=catalog_ext, print_catalog=False)
    else :
        # get catalog
        catalog = get_sci_image_catalog(s4_proc_frame, catalog_ext=catalog_ext, print_catalog=False, wcs=wcs)

    #catalog.write("catalog.csv",overwrite=True)
   
    """
    img_data = fits.getdata(s4_proc_frame,0)
    new_sky_coords = np.ndarray((len(catalog["RA"]), 2))
    for j in range(len(catalog["RA"])) :
        new_sky_coords[j] = [catalog["RA"][j],catalog["DEC"][j]]
    cat_sources_pixcoords = np.array(wcs.world_to_pixel_values(new_sky_coords))
    plt.imshow(img_data, vmin=np.median(img_data), vmax=2 * np.median(img_data), cmap="Greys_r")
    _ = photutils.aperture.CircularAperture(cat_sources_pixcoords, r=8.0).plot(color="g")
    plt.show()
    """
    
    # get the center of the image
    ra_cen, dec_cen = wcs.wcs.crval[0], wcs.wcs.crval[1]
    # set image center coordinates
    center = SkyCoord(ra_cen, dec_cen, unit=(u.deg,u.deg), frame='icrs')
    #print(center.ra.hms, center.dec.dms)
    
    vizier_query_result = query_vizier_catalog(center, search_radius=search_radius, catalogname=vizier_catalog_name, max_mag=max_mag, max_sources=max_sources)
    
    if vizier_query_result is None :
        print("WARNING: could not match sources in Vizier, returning unchanged catalog")
        return catalog
    
    vizier_sources, cat_sources = Table(names=vizier_query_result.colnames), []
    vizier_matched_sources_indexes = []
    
    for j in range(len(catalog)) :
    
        cat_sources.append(catalog[j])
        
        cat_ra, cat_dec = catalog["RA"][j], catalog["DEC"][j]
        
        delta_min, imin  = 1e30, np.nan
        
        for i in range(len(vizier_query_result)) :
            ra_i, dec_i = vizier_query_result['ra_deg'][i], vizier_query_result['dec_deg'][i]
                        
            dra = np.cos(dec_i*np.pi/180.)*(cat_ra - ra_i)*60*60
            ddec = (cat_dec - dec_i)*60*60
            delta = np.sqrt((dra)**2 + (ddec)**2)

            if delta < delta_min and delta < match_threshold :
                delta_min = delta
                imin = i
    
        if np.isfinite(imin) :
            vizier_sources.add_row(vizier_query_result[imin])
            vizier_matched_sources_indexes.append(j)
        else :
            vizier_sources.add_row(None)

    n_vizier_matched_sources = len(vizier_matched_sources_indexes)
    print("Number of matched sources with Vizier: {}".format(n_vizier_matched_sources))
    
    img_data = fits.getdata(s4_proc_frame,0)
    vizier_sources_skycoords = []
    cat_sources_skycoords = []

    for i in vizier_matched_sources_indexes :
        vizier_ra, vizier_dec = vizier_sources[i]['ra_deg'], vizier_sources[i]['dec_deg']
        cat_ra, cat_dec = cat_sources[i]['RA'], cat_sources[i]['DEC']
        vizier_sources_skycoords.append([vizier_ra, vizier_dec])
        cat_sources_skycoords.append([cat_ra, cat_dec])
            
    vizier_sources_skycoords = np.array(vizier_sources_skycoords,dtype=float)
    cat_sources_skycoords = np.array(cat_sources_skycoords,dtype=float)
    
    vizier_sources_pixcoords = np.array(wcs.world_to_pixel_values(vizier_sources_skycoords))
    
    cat_sources_pixcoords = np.array(wcs.world_to_pixel_values(cat_sources_skycoords))
    if plot :
        plt.imshow(img_data, vmin=np.median(img_data), vmax=2 * np.median(img_data), cmap="Greys_r")
        _ = photutils.aperture.CircularAperture(vizier_sources_pixcoords, r=10.0).plot(color="r")
        _ = photutils.aperture.CircularAperture(cat_sources_pixcoords, r=8.0).plot(color="g")
        plt.show()
    
    catalog.rename_column('RA', 'OBS_RA')
    catalog.rename_column('DEC', 'OBS_DEC')
    
    for colname in vizier_sources.colnames :
        if colname == 'ident' :
            catalog.add_column(vizier_sources[colname], index=0)
        else :
            catalog.add_column(vizier_sources[colname])
    catalog.rename_column('ident', '{} ID'.format(vizier_catalog_name))
    catalog.add_column(vizier_sources['ident'], index=0)

    return catalog
    
    
def query_vizier_catalog(center, search_radius=5.6, catalogname='SkyMapper', max_mag=21, max_sources=1e4) :

    """ Module to query a Vizier catalog
    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        center coordinates
    search_radius : float (optional)
        search radius around center of the field [arcmin]
    catalogname : str (optional)
        Vizier catalog name. Supported catalogs: 'SkyMapper', 'APASS9', or 'GAIADR2'
    max_mag : float (optional)
        maximum magnitude to consider in the matched catalog
    max_sources : int (optional)
        maximum number of sources to truncate search
    Returns
        catalog: astropy.table.Table
            updated catalog of sources
    -------
    """
    data = None

    rad_arcmin = search_radius * u.arcminute

    if catalogname == 'SkyMapper':
        vquery = Vizier(columns=['all'],column_filters={"rPSF":("<{:f}".format(max_mag))},row_limit=max_sources,timeout=300)
        
        try:
            data = vquery.query_region(center, radius=rad_arcmin, catalog="II/358/smss",cache=False)[0]
        except :
            print('no data available from {:s}'.format(catalogname))
            return data
        # throw out columns we don't need
        new_data = Table(data)
        for col in data.columns:
            if col not in ['ObjectId', 'RAICRS', 'DEICRS',
                            'e_RAICRS', 'e_DEICRS',
                            'uPSF', 'e_uPSF',
                            'vPSF', 'e_vPSF',
                            'gPSF', 'e_gPSF',
                            'rPSF', 'e_rPSF',
                            'zPSF', 'e_zPSF',
                            'iPSF', 'e_iPSF']:
                new_data.remove_column(col)
        data = new_data

        # rename column names using PP conventions
        data.rename_column('ObjectId', 'ident')
        data.rename_column('RAICRS', 'ra_deg')
        data.rename_column('DEICRS', 'dec_deg')
        data.rename_column('e_RAICRS', 'e_ra_deg')
        data['e_ra_deg'].convert_unit_to(u.deg)
        data.rename_column('e_DEICRS', 'e_dec_deg')
        data['e_dec_deg'].convert_unit_to(u.deg)
        #data.rename_column('uPSF', 'umag')
        #data.rename_column('e_uPSF', 'e_umag')
        data.rename_column('vPSF', 'vsmmag')
        data.rename_column('e_vPSF', 'e_vsmmag')
        data.rename_column('gPSF', 'gsmmag')
        data.rename_column('e_gPSF', 'e_gsmmag')
        data.rename_column('rPSF', 'rsmmag')
        data.rename_column('e_rPSF', 'e_rsmmag')
        data.rename_column('iPSF', 'ismmag')
        data.rename_column('e_iPSF', 'e_ismmag')
        data.rename_column('zPSF', 'zsmmag')
        data.rename_column('e_zPSF', 'e_zsmmag')

        data = data[data['e_rsmmag'] <= 0.03]


    elif catalogname == 'APASS9':
        # photometric catalog
        vquery = Vizier(columns=['recno', 'RAJ2000', 'DEJ2000',
                                 'e_RAJ2000',
                                 'e_DEJ2000', 'Vmag', 'e_Vmag',
                                 'Bmag', 'e_Bmag', "g'mag", "e_g'mag",
                                 "r'mag", "e_r'mag", "i'mag", "e_i'mag"],
                                 column_filters={"Vmag":("<{:f}".format(max_mag))},
                                 row_limit=max_sources)

        try:
            data = vquery.query_region(center, radius=rad_arcmin,
                                                catalog="II/336/apass9",
                                                cache=False)[0]
        except :
            print('no data available from {:s}'.format(catalogname))
            return data

        # rename column names using PP conventions
        data.rename_column('recno', 'ident')
        data.rename_column('RAJ2000', 'ra_deg')
        data.rename_column('DEJ2000', 'dec_deg')
        data.rename_column('e_RAJ2000', 'e_ra_deg')
        data.rename_column('e_DEJ2000', 'e_dec_deg')
        data.rename_column('g_mag', 'gmag')
        data.rename_column('e_g_mag', 'e_gmag')
        data.rename_column('r_mag', 'rmag')
        data.rename_column('e_r_mag', 'e_rmag')
        data.rename_column('i_mag', 'imag')
        data.rename_column('e_i_mag', 'e_imag')

    elif catalogname == 'GAIADR2':
        # astrometric and photometric catalog (as of DR2)
        vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS',
                                 'e_RA_ICRS', 'e_DE_ICRS', 'pmRA',
                                 'pmDE', 'Epoch',
                                 'Gmag', 'e_Gmag',
                                 'BPmag', 'e_BPmag',
                                 'RPmag', 'eRPmag'],
                                 column_filters={"phot_g_mean_mag":("<{:f}".format(max_mag))},
                                 row_limit=max_sources,timeout=300)

        try:
            data = vquery.query_region(center, radius=rad_arcmin,
                                       catalog="I/345/gaia2",
                                       cache=False)[0]
        except IndexError :
            print('no data available from {:s}'.format(catalogname))
            return data

        # rename column names using PP conventions
        data.rename_column('Source', 'ident')
        data.rename_column('RA_ICRS', 'ra_deg')
        data.rename_column('DE_ICRS', 'dec_deg')
        data.rename_column('e_RA_ICRS', 'e_ra_deg')
        data['e_ra_deg'].convert_unit_to(u.deg)
        data.rename_column('e_DE_ICRS', 'e_dec_deg')
        data['e_dec_deg'].convert_unit_to(u.deg)
        data.rename_column('Epoch', 'epoch_yr')
        data['mag'] = data['Gmag']  # required for scamp


    return data


def transform_SkyMapper_to_Sloan_griz(data) :

    """ Module to transform magnitudes to SDSS AB system
        using linear transformation equations derived from
        data in Wolf et al. 2018 PASA Vol 35 uncertainties
        from linear fit
    Parameters
    ----------
    data: astropy.table.Table
        input catalog of sources
    Returns
        data: astropy.table.Table
            updated catalog of sources
    -------
    """

    g = data['gsmmag'].data
    e_g = data['e_gsmmag'].data
    r = data['rsmmag'].data
    e_r = data['e_rsmmag'].data
    i = data['ismmag'].data
    e_i = data['e_ismmag'].data
    z = data['zsmmag'].data
    e_z = data['e_zsmmag'].data

    #g mag
    g_sdss = np.zeros(g.shape[0])
    gerr_sdss = np.zeros(g.shape[0])
            
    for ism in range(g.shape[0]):
        if (g[ism]-i[ism]) <  1.5:
            g_sdss[ism] = g[ism] - (-0.2366)*(g[ism]-i[ism]) - (-0.0598)
            gerr_sdss[ism] = np.sqrt(e_g[ism]**2 + 0.0045**2 + 0.0036**2)
        elif (g[ism]-i[ism]) > 1.5:
            g_sdss[ism] = g[ism] - (0.1085)*(g[ism]-i[ism]) - (-0.5700)
            gerr_sdss[ism] = np.sqrt(e_g[ism]**2 + 0.0148**2 + 0.0399**2)
            
    #r mag
    r_sdss = np.zeros(r.shape[0])
    rerr_sdss = np.zeros(r.shape[0])
            
    for ism in range(r.shape[0]):
        if (g[ism]-i[ism]) <  1.5:
            r_sdss[ism] = r[ism] - (0.0318)*(g[ism]-i[ism]) - (0.0011)
            rerr_sdss[ism] = np.sqrt(e_r[ism]**2 + 0.0009**2 + 0.0007**2)
        elif (g[ism]-i[ism]) > 1.5:
            r_sdss[ism] = r[ism] - (-0.0472)*(g[ism]-i[ism]) - (0.1127)
            rerr_sdss[ism] = np.sqrt(e_r[ism]**2 + 0.0062**2 + 0.0168**2)
    
    #i mag
    i_sdss = np.zeros(i.shape[0])
    ierr_sdss = np.zeros(i.shape[0])
            
    for ism in range(i.shape[0]):
        if (g[ism]-i[ism]) <  1.1:
            i_sdss[ism] = i[ism] - (-0.0389)*(g[ism]-i[ism]) - (0.0137)
            ierr_sdss[ism] = np.sqrt(e_i[ism]**2 + 0.0023**2 + 0.0015**2)
        elif (g[ism]-i[ism]) > 1.1:
            i_sdss[ism] = i[ism] - (-0.0617)*(g[ism]-i[ism]) - (0.0362)
            ierr_sdss[ism] = np.sqrt(e_i[ism]**2 + 0.0020**2 + 0.0048**2)
            
    #z mag
    z_sdss = np.zeros(z.shape[0])
    zerr_sdss = np.zeros(z.shape[0])
        
    for ism in range(z.shape[0]):
        if (g[ism]-i[ism]) <  2.0:
            z_sdss[ism] = z[ism] - (-0.0352)*(g[ism]-i[ism]) - (0.0095)
            zerr_sdss[ism] = np.sqrt(e_z[ism]**2 + 0.0017**2 + 0.0018**2)
        elif (g[ism]-i[ism]) > 2.0:
            z_sdss[ism] = z[ism] - (-0.1209)*(g[ism]-i[ism]) - (0.2115)
            zerr_sdss[ism] = np.sqrt(e_z[ism]**2 + 0.0155**2 + 0.0470**2)
            
    data.add_column(Column(data=g_sdss, name='_gmag',unit=u.mag))
    data.add_column(Column(data=gerr_sdss, name='_e_gmag',unit=u.mag))
    data.add_column(Column(data=r_sdss, name='_rmag',unit=u.mag))
    data.add_column(Column(data=rerr_sdss, name='_e_rmag',unit=u.mag))
    data.add_column(Column(data=i_sdss, name='_imag',unit=u.mag))
    data.add_column(Column(data=ierr_sdss, name='_e_imag',unit=u.mag))
    data.add_column(Column(data=z_sdss, name='_zmag',unit=u.mag))
    data.add_column(Column(data=zerr_sdss, name='_e_zmag',unit=u.mag))
            
    # get rid of sources with no information from catalog
    data = data[data['_gmag'] != 0]
    data = data[data['_e_gmag'] != 0]
    data = data[data['_rmag'] != 0]
    data = data[data['_e_rmag'] != 0]
    data = data[data['_imag'] != 0]
    data = data[data['_e_imag'] != 0]
    data = data[data['_zmag'] != 0]
    data = data[data['_e_zmag'] != 0]
    
    return data


def transform_GaiaDR2_to_Sloan_griz(data, apply_color_mask=True) :

    """ Module to transform magnitudes from
        Gaia DR2+ to SDSS AB system using
        http://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
    Parameters
    ----------
    data: astropy.table.Table
        input catalog of sources
    apply_color_mask: bool
        to select sources applying a color mask
    Returns
        data: astropy.table.Table
            updated catalog of sources
    -------
    """

    if apply_color_mask :
        colormask = ((data['BPmag']-data['RPmag'] > -0.5) &
                     (data['BPmag']-data['RPmag'] < 2.0))
                     
        colormask &= ((data['BPmag']-data['RPmag'] > 0.2) &
                      (data['BPmag']-data['RPmag'] < 2.7))
        
        colormask &= ((data['BPmag']-data['RPmag'] > 0) &
                      (data['BPmag']-data['RPmag'] < 4.5))

        data = data[colormask]

    g = data['Gmag'].data
    e_g = data['e_Gmag'].data
    bp = data['BPmag'].data
    rp = data['RPmag'].data

    g_sdss = g - (0.13518 - 0.46245*(bp-rp) - 0.25171*(bp-rp)**2 + 0.021349*(bp-rp)**3)
    e_g_sdss = np.sqrt(e_g**2 + 0.16497**2)
    r_sdss = g - (-0.12879 + 0.24662*(bp-rp) - 0.027464*(bp-rp)**2 - 0.049465*(bp-rp)**3)
    e_r_sdss = np.sqrt(e_g**2 + 0.066739**2)
    i_sdss = g - (-0.29676 + 0.64728*(bp-rp) - 0.10141*(bp-rp)**2)
    e_i_sdss = np.sqrt(e_g**2 + 0.098957**2)

    data.add_column(Column(data=g_sdss, name='_gmag',unit=u.mag))
    data.add_column(Column(data=e_g_sdss, name='_e_gmag',unit=u.mag))
    data.add_column(Column(data=r_sdss, name='_rmag',unit=u.mag))
    data.add_column(Column(data=e_r_sdss, name='_e_rmag',unit=u.mag))
    data.add_column(Column(data=i_sdss, name='_imag',unit=u.mag))
    data.add_column(Column(data=e_i_sdss, name='_e_imag',unit=u.mag))

    return data


def odd_ratio_mean(value, err, odd_ratio=1e-4, nmax=10):

    """ Module to calculate odd ratio mean
    
    Parameters
    ----------
    data: np.array()
        input data array of floats
    err: np.array()
        input err array of floats
    odd_ratio: float
        odd ration threshold
    max: int
        maximum number of iterations
        
    Returns
        guess, guess_err: float, float
        mean value and uncertainty
    -------
    """
    keep = np.isfinite(value)*np.isfinite(err)
    
    if np.sum(keep) == 0:
        return np.nan, np.nan

    value = value[keep]
    err = err[keep]

    guess = np.nanmedian(value)
    guess_err = np.nanmedian(np.abs(value - guess)) / 0.67449

    nite = 0
    while (nite < nmax):
        nsig = (value-guess)/np.sqrt(err**2 + guess_err**2)
        gg = np.exp(-0.5*nsig**2)
        odd_bad = odd_ratio/(gg+odd_ratio)
        odd_good = 1-odd_bad
        
        w = odd_good/(err**2 + guess_err**2)
        
        guess = np.nansum(value*w)/np.nansum(w)
        guess_err = np.sqrt(1/np.nansum(w))
        nite+=1

    return guess, guess_err


def bin_data(x, y, yerr, median=False, binsize = 0.005, min_npoints=1) :

    """ Module to bin data
    
    Parameters
    ----------
    x: np.array()
        input array of floats
    y: np.array()
        input array of floats
    yerr: float
        nput array of floats
    median: bool
        to calculate each bin by the median
    binsize: float
        bin size in the same units as the x array
    min_points: int
        minimum number of points to consider within each bin
        
    Returns
        bin_x, bin_y, bin_yerr: np.array(),np.array(),np.array()
        binned data
    -------
    """
    
    xi, xf = x[0], x[-1]
    
    bins = np.arange(xi, xf, binsize)
    digitized = np.digitize(x, bins)
    
    bin_y, bin_yerr = [], []
    bin_x = []
    
    for i in range(0,len(bins)+1):
        if len(x[digitized == i]) > min_npoints:
            if median :
                mean_y = np.nanmedian(y[digitized == i])
                myerr = np.nanmedian(np.abs(y[digitized == i] - mean_y))  / 0.67449
                #weights = 1/(yerr[digitized == i]**2)
                #mean_y = np.average(y[digitized == i], weights=weights)
                #myerr = np.sqrt(np.sum((weights**2)*(yerr[digitized == i]**2)))
            else :
                mean_y, myerr = odd_ratio_mean(y[digitized == i], yerr[digitized == i])
                #mean_y = np.nanmean(y[digitized == i])
                #myerr = np.nanstd(y[digitized == i])
            
            bin_x.append(np.nanmean(x[digitized == i]))
            bin_y.append(mean_y)
            bin_yerr.append(myerr)

    bin_y, bin_yerr = np.array(bin_y), np.array(bin_yerr)
    bin_x = np.array(bin_x)

    return bin_x, bin_y, bin_yerr


def linear_fit_to_photometry(mag, emag, obs_mag, obs_emag, nstd = 3, band='', catalog_name='', plot=False, verbose=False) :

    """ Module to perform a least-squares linear fit to photometry data
    
    Parameters
    ----------
    mag: np.array()
        input array of catalog magnitudes
    emag: np.array()
        input array of catalog magnitudes errors
    obs_mag: float
        input array of observed magnitudes
    obs_emag: float
        input array of observed magnitudes errors
    median: bool
        to calculate each bin by the median
    nstd: int
        number of sigma for clipping
    band: str
        spectral band id
    catalog_name: str
        catalog name
    plot: bool
        to plot results
    verbose: bool
        to print verbose messages
        
    Returns
        a, b : ufloat, ufloat
        slope and intercept of linear solution
    -------
    """


    def func(p, x):
        a, b = p
        return a*x + b

    # Model object
    lin_model = odr.Model(func)

    # Create a RealData object
    data = odr.RealData(mag, obs_mag, sx=emag, sy=obs_emag)
        
    # Set up ODR with the model and data.
    myodr = odr.ODR(data, lin_model, beta0=[0., 1.])
        
    # Run the regression.
    results = myodr.run()
    
    #print fit parameters and 1-sigma estimates
    popt = results.beta
    perr = results.sd_beta
    print('fit parameter 1-sigma error')
    if verbose:
        for i in range(len(popt)):
            print(str(popt[i])+' +- '+str(perr[i]))
        
    # prepare confidence level curves
    # to draw nstd-sigma intervals
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    x_fit = np.linspace(np.min(mag), np.max(mag), 100)
    fit = func(popt, x_fit)
    fit_up = func(popt_up, x_fit)
    fit_dw = func(popt_dw, x_fit)
    
    a = ufloat(popt[1],perr[1])
    b = ufloat(popt[0],perr[0])

    resids = obs_mag - func(popt, mag)
    print("Photometric fit RMS = {:.5f}".format(np.std(np.abs(resids))))
    
    if plot :
        plt.title("{}-band phot calib from {} catalog".format(band,catalog_name))
        plt.plot(x_fit, fit*0., 'r', lw=2, label="f({}') = {} + {}*{}'".format(band,a,b,band))
        #plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label= '3-sigma interval')
        #plt.errorbar(mag,obs_mag,xerr=emag,yerr=obs_emag, fmt='o', ecolor='k')
        plt.errorbar(mag,resids,xerr=emag,yerr=obs_emag, fmt='o', ecolor='k')
        plt.ylabel(r"{}$_i$ - f({}') [mag]".format(band,band), fontsize=18)
        plt.xlabel(r"{}' [mag]".format(band), fontsize=18)
        plt.legend(fontsize=14)
        plt.show()
    
    return a, b
    

def calibrate_photometry_in_stack_catalog(objprods, aperture_radius=10, vizier_catalog_name='SkyMapper', dmag_sig_clip=1.0, plot=False, pol_beam="N") :

    """ Module to match catalog sources with sources of a given Vizier catalog
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    vizier_catalog_name : str (optional)
        Vizier catalog name
    dmag_sig_clip : float
        Threshold for excluding sources based on the magnitude difference from catalog values.
    plot : bool
        to plot results
    Returns
        catalogs, objprods: list: [Table, Table, Table, Table], dict
            calibrated catalogs and calibration data saved in the objprods container
    -------
    """

    bands = ['g', 'r', 'i', 'z']

    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    if objprods["instmode"] == "POLAR" :
        catalog_ext = "CATALOG_POL_{}_AP{:03d}".format(pol_beam,aperture_radius)
             
    objprods["phot_calibration"] = [None, None, None, None]
    catalogs = [None, None, None, None]
    
    for ch in range(4) :
    
        catalog_match = match_catalog_sources_with_vizier(objprods["stack"][ch], wcs=objprods["astrometry_calibration"][ch], catalog_ext=catalog_ext, search_radius=5, match_threshold=2., vizier_catalog_name=vizier_catalog_name, plot=plot)
            
        #print("len(catalog_match)=",len(catalog_match))
        #catalog_match.write("SkyMapper_crab_{}-band.csv".format(bands[ch]),overwrite=True)
        
        if vizier_catalog_name == 'SkyMapper' :
            if 'gsmmag' not in catalog_match.colnames :
                return catalogs, objprods
            catalog_match = transform_SkyMapper_to_Sloan_griz(catalog_match)
                
        elif vizier_catalog_name == 'GAIADR2' :
            if 'Gmag' not in catalog_match.colnames or ch > 2:
                return catalogs, objprods
            catalog_match = transform_GaiaDR2_to_Sloan_griz(catalog_match, apply_color_mask=True)
    
        else :
            return catalogs, objprods
            
        src_idx = catalog_match['SRCINDEX'].data
        obs_mag = catalog_match['MAG'].data
        obs_emag = catalog_match['EMAG'].data
        mag = catalog_match['_{}mag'.format(bands[ch])].data
        emag = catalog_match['_e_{}mag'.format(bands[ch])].data

        #print(catalog)
        #print(type(mag), np.shape(mag))
        #print(type(emag), np.shape(emag))
        #print(len(mag), len(emag), len(obs_mag), len(obs_emag))

        if vizier_catalog_name == 'SkyMapper' :
            orig_mag = catalog_match['{}smmag'.format(bands[ch])].data
        elif vizier_catalog_name == 'GAIADR2' :
            orig_mag = catalog_match['Gmag'].data

        keep = orig_mag != 0
        keep &= (np.isfinite(obs_mag)) & (np.isfinite(orig_mag))
        keep &= (np.isfinite(mag)) & (np.isfinite(emag))

        if len(mag[keep]) > 1 :
            a, b = linear_fit_to_photometry(mag[keep], emag[keep], obs_mag[keep], obs_emag[keep], nstd = 3, band=bands[ch], catalog_name=vizier_catalog_name, plot=plot, verbose=False)
        else :
            print("WARNING: could not calculate photometric calibration for channel={}",ch+1)
            a,b=0,1

        calib_mag, calib_emag = np.zeros_like(mag), np.zeros_like(mag)
        for i in range(len(mag)) :
            new_mag = (ufloat(obs_mag[i],obs_emag[i]) - a) / b
            calib_mag[i] = new_mag.nominal_value
            calib_emag[i] = new_mag.std_dev

        catalog = get_sci_image_catalog(objprods["stack"][ch], catalog_ext=catalog_ext, wcs=objprods["astrometry_calibration"][ch])
            
        idx = catalog['SRCINDEX'].data
        outmag, outemag = np.full_like(idx,0), np.full_like(idx,0)
        outcalib_mag, outcalib_emag = np.full_like(idx,0), np.full_like(idx,0)
        for i in range(len(idx)) :
            for j in range(len(src_idx)) :
                if idx[i] == src_idx[j] and np.isfinite(mag[j]) and np.isfinite(calib_mag[j]):
                    if np.abs(calib_mag[j] - mag[j]) < dmag_sig_clip :
                        outmag[i] = mag[j]
                        outemag[i] = emag[j]
                        outcalib_mag[i] = calib_mag[j]
                        outcalib_emag[i] = calib_emag[j]
            
        catalog.add_column(Column(data=outmag, name='{}_mag'.format(bands[ch]),unit=u.mag))
        catalog.add_column(Column(data=outemag, name='{}_emag'.format(bands[ch]),unit=u.mag))

        catalog.add_column(Column(data=outcalib_mag, name='_{}mag'.format(bands[ch]),unit=u.mag))
        catalog.add_column(Column(data=outcalib_emag, name='_e_{}mag'.format(bands[ch]),unit=u.mag))

        catalogs[ch] = catalog
            
        objprods["phot_calibration"][ch] = [a,b]
            
    return catalogs, objprods
    

def create_master_catalog() :

    """ Module to create a master catalog
    Parameters
    ----------
    Returns
        tbl: astropy.table.Table
            empty master catalog
    -------
    """


    tbl = Table()
        
    id_colnames = ['gaiadr2_id','srcindex_g', 'srcindex_r', 'srcindex_i', 'srcindex_z']
    skycoord_colnames = ['ra', 'dec', 'e_ra', 'e_dec']
    pm_colnames = ['pmRA', 'pmDE']
    epoch_colname = 'epoch_yr'
    mag_colnames = ['mag_g', 'emag_g',
                    'mag_r', 'emag_r',
                    'mag_i', 'emag_i',
                    'mag_z', 'emag_z',
                    '_gmag', '_e_gmag',
                    '_rmag', '_e_rmag',
                    '_imag', '_e_imag',
                    '_zmag', '_e_zmag']
 
    for col in id_colnames :
        tbl.add_column(Column(data=np.array([]), name=col))

    for col in skycoord_colnames :
        tbl.add_column(Column(data=np.array([]), name=col,unit=u.deg))
        
    for col in pm_colnames :
        tbl.add_column(Column(data=np.array([]), name=col,unit=u.mas/u.yr))

    tbl.add_column(Column(data=np.array([]), name=epoch_colname,unit=u.yr))
    
    for col in mag_colnames :
        tbl.add_column(Column(data=np.array([]), name=col,unit=u.mag))
    
    return tbl
    

def master_catalog_in_four_bands(objprods, calibrate_photometry=False, aperture_radius=10, pol_beam="N") :

    """ Module to create a master catalogs containing photometric data in all 4 sparc4 bands
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    calibrate_photometry : bool (optional)
        to calibrate photometry
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    Returns
        master_catalog: astropy.table.Table
            master catalog
    -------
    """

    bands = ['g', 'r', 'i', 'z']
    
    master_catalog = create_master_catalog()

    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    if objprods["instmode"] == "POLAR" :
        catalog_ext = "CATALOG_POL_{}_AP{:03d}".format(pol_beam,aperture_radius)
             
    # all channels starting by the one with highest throughput
    for ch in range(4) :
    
        if objprods["stack"][ch] is None:
            continue
        
        catalog_gaiadr2_match = match_catalog_sources_with_vizier(objprods["stack"][ch], wcs=objprods["astrometry_calibration"][ch], catalog_ext=catalog_ext, search_radius=5, vizier_catalog_name='GAIADR2')
        catalog_gaiadr2_match = transform_GaiaDR2_to_Sloan_griz(catalog_gaiadr2_match, apply_color_mask=True)
            
        catalog_skymapper_match = match_catalog_sources_with_vizier(objprods["stack"][ch], wcs=objprods["astrometry_calibration"][ch], catalog_ext=catalog_ext, search_radius=5, vizier_catalog_name='SkyMapper')
        if 'gsmmag' in catalog_skymapper_match.colnames :
            catalog_skymapper_match = transform_SkyMapper_to_Sloan_griz(catalog_skymapper_match)
            
        src_index = catalog_gaiadr2_match['SRCINDEX'].data
        gaiadr2id = catalog_gaiadr2_match['GAIADR2 ID'].data

        mag_key = '_{}mag'.format(bands[ch])
        emag_key = '_e_{}mag'.format(bands[ch])
                
        try :
            mag = catalog_skymapper_match[mag_key].data
            emag = catalog_skymapper_match[emag_key].data
        except :
            print("WARNING: could not match with skymapper, using GAIA converted magnitudes")
            mag = catalog_gaiadr2_match[mag_key].data
            emag = catalog_gaiadr2_match[emag_key].data
                    
        obs_mag = catalog_gaiadr2_match['MAG'].data
        obs_emag = catalog_gaiadr2_match['EMAG'].data

        ra = catalog_gaiadr2_match['ra_deg'].data
        dec = catalog_gaiadr2_match['dec_deg'].data
        era = catalog_gaiadr2_match['e_ra_deg'].data
        edec = catalog_gaiadr2_match['e_dec_deg'].data

        pmRA = catalog_gaiadr2_match['pmRA'].data
        pmDE = catalog_gaiadr2_match['pmDE'].data
        epoch_yr = catalog_gaiadr2_match['epoch_yr'].data
            
        a, b = 0., 1.
        if  calibrate_photometry and objprods["phot_calibration"][ch] is not None :
            a, b = objprods["phot_calibration"][ch]
                
        for i in range(len(catalog_gaiadr2_match['MAG'])) :
            
            new_mag = (ufloat(obs_mag[i],obs_emag[i]) - a) / b
                  
            if  gaiadr2id[i] in master_catalog['gaiadr2_id'].data :
                idx = np.nonzero(master_catalog['gaiadr2_id'] == gaiadr2id[i])[0][0]
            else :
                idx = len(master_catalog)
                master_catalog.add_row()
                    
                for j in range(4) :
                    master_catalog['srcindex_{}'.format(bands[j])][idx] = None

                master_catalog['gaiadr2_id'][idx] = gaiadr2id[i]
                
                master_catalog['ra'][idx] = ra[i]
                master_catalog['dec'][idx] = dec[i]
                master_catalog['e_ra'][idx] = era[i]
                master_catalog['e_dec'][idx] = edec[i]

                master_catalog['pmRA'][idx] = pmRA[i]
                master_catalog['pmDE'][idx] = pmDE[i]

                master_catalog['epoch_yr'][idx] = epoch_yr[i]
                
            #print("{}-band ID={} i={} src_index={} idx={} mag={} _mag={}".format(bands[ch], gaiadr2id[i], i, src_index[i], idx, mag[i], new_mag.nominal_value))
            master_catalog['srcindex_{}'.format(bands[ch])][idx] = int(src_index[i])
                    
            master_catalog['mag_{}'.format(bands[ch])][idx] = new_mag.nominal_value
            master_catalog['emag_{}'.format(bands[ch])][idx] = new_mag.std_dev

            master_catalog['_{}mag'.format(bands[ch])][idx] = mag[i]
            master_catalog['_e_{}mag'.format(bands[ch])][idx] = emag[i]
            
            
    return master_catalog


def select_comparison_stars(master_catalog, target_indexes=[0,0,0,0], comps1="", comps2="", comps3="", comps4="", max_mag_diff=2) :

    """ Module to select comparison stars automatically
    
    Parameters
    ----------
    master_catalog: astropy.table.Table
        master catalog
    target_indexes: list: [int, int, int, int]
        list of target indexes
    comps1 : str
        comparison indexes for channel 1, separated by comma. eg. "2,3,4"
    comps2 : str
        comparison indexes for channel 2, separated by comma. eg. "2,3,4"
    comps3 : str
        comparison indexes for channel 3, separated by comma. eg. "2,3,4"
    comps4 : str
        comparison indexes for channel 4, separated by comma. eg. "2,3,4"
    max_mag_diff: float
        maximum magnitude difference betweeen target and selected comparison stars
    Returns
        comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
            list of arrays of comparison indexes in the four channels
    -------
    """

    comps = []
    
    if comps1 != "" :
        comps1_idx = comps1.split(",")
        comps1_indexes = []
        for i in range(len(comps1_idx)) :
            comps1_indexes.append(int(comps1_idx[i]))
        comps.append(comps1_indexes)
    else :
        comps.append([])
    
    if comps2 != "" :
        comps2_idx = comps2.split(",")
        comps2_indexes = []
        for i in range(len(comps2_idx)) :
            comps2_indexes.append(int(comps2_idx[i]))
        comps.append(comps2_indexes)
    else :
        comps.append([])
           
    if comps3 != "" :
        comps3_idx = comps3.split(",")
        comps3_indexes = []
        for i in range(len(comps3_idx)) :
            comps3_indexes.append(int(comps3_idx[i]))
        comps.append(comps3_indexes)
    else :
        comps.append([])
        
    if comps4 != "" :
        comps4_idx = comps4.split(",")
        comps4_indexes = []
        for i in range(len(comps4_idx)) :
            comps4_indexes.append(int(comps4_idx[i]))
        comps.append(comps4_indexes)
    else :
        comps.append([])
    
    bands = ['g', 'r', 'i', 'z']

    for ch in range(4) :
        if len(comps[ch]) == 0 :
            for i in range(len(master_catalog)) :
                if i != target_indexes[ch] :
                    mag = master_catalog['mag_{}'.format(bands[ch])][target_indexes[ch]]
                    comp_mag = master_catalog['mag_{}'.format(bands[ch])][i]
                    currcompindex = master_catalog['srcindex_{}'.format(bands[ch])][i]
            
                    if np.abs(comp_mag - mag) < max_mag_diff and np.isfinite(currcompindex) :
                        comps[ch].append(int(currcompindex))
    return comps



def differential_light_curves(objprods, object_indexes, comps, binsize=0.003, aperture_radius=10, object_name="", plot=False) :


    """ Module to get differential light curves and auxiliary data
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    object_indexes: list: [int, int, int, int]
        list of target indexes
    comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
        list of arrays of comparison indexes in the four channels
    binsize : float
        time bin size in units of days
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    plot : bool
        to plot
    
    Returns
        lcs: list
            set of 4 lightcurve data
    -------
    """

    lcs = []

    bands = ['g', 'r', 'i', 'z']
    colors = ['darkblue','darkgreen','darkorange','darkred']
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)

    if plot :
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        
    for ch in range(4) :
        
        if ch == 0 and plot :
            axs[ch].set_title(r"{}".format(object_name), fontsize=20)

        try :
            lc = s4plt.plot_light_curve(objprods["lc"][ch], target=object_indexes[ch], comps=comps[ch], catalog_name=catalog_ext, plot_coords=False, plot_rawmags=False, plot_sum=False, plot_comps=False)
    
            x, y, yerr = bin_data(lc['TIME'], lc['diffmagsum'], lc['magsum_err'], median=False, binsize = binsize)
                
            keep = lc['TIME'] > 2460561.74
            keepbin = x > 2460561.74

            sampling = np.median(np.abs(lc['TIME'][1:] - lc['TIME'][:-1])) * 24 * 60 * 60
            samplingbin = np.median(np.abs(x[1:] - x[:-1])) * 24 * 60

            print("RMS = {:.6f}  Sampling={:.1f} s".format(np.nanstd(lc['diffmagsum'][keep]),sampling))
            print("RMS (binned) = {:.6f} Sampling={:.2f} min".format(np.nanstd(y[keepbin]),samplingbin))
    
            if plot :
                axs[ch].errorbar(x,y,yerr=yerr,fmt="o",color='k')
                axs[ch].plot(lc['TIME'],lc['diffmagsum'],".",color=colors[ch], alpha=0.3, label=bands[ch])
            lcs.append(lc)
        except :
            lcs.append(None)
            continue
        
        if plot :
            axs[ch].tick_params(axis='x', labelsize=14)
            axs[ch].tick_params(axis='y', labelsize=14)
            axs[ch].minorticks_on()
            axs[ch].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
            axs[ch].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
            if ch == 3 :
                axs[ch].set_xlabel(r"time (BJD)", fontsize=16)
                
            axs[ch].set_ylabel(r"$\Delta$mag", fontsize=16)
            axs[ch].legend(fontsize=16)

    if plot :
        plt.show()
        
    return lcs


def get_polarimetry_time_series(objprods, object_indexes, comps, binsize=0.003, aperture_radius=10, plot_total_polarization=True, plot_theta=False, plot_u=False, object_name="", plot=False) :

    """ Module to get polarimetry time series
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis
    object_indexes: list: [int, int, int, int]
        list of target indexes
    comps: list: [list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..], list:[c1, c2, c3, ..]]
        list of arrays of comparison indexes in the four channels
    binsize : float
        time bin size in units of days
    aperture_radius : int (optional)
        photometry aperture radius (pixels) to select FITS hdu extension in catalog
    object_name : str
        object name
    
    Returns
        lcs: list
            set of 4 lightcurve data
    -------
    """

    lcs = []

    bands = ['g', 'r', 'i', 'z']
    colors = ['darkblue','darkgreen','darkorange','darkred']
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
          
    if plot :
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0].set_title(r"{}".format(object_name), fontsize=20)
        
    for ch in range(4) :
        
        try :
            if plot_total_polarization is False :
                plot_theta = False
        
            ts = s4plt.plot_polar_time_series(objprods["ts"][ch], target=object_indexes[ch], comps=comps[ch], nsig=10, plot_total_polarization=plot_total_polarization, plot_polarization_angle=plot_theta, plot_comps=True, plot_sum=True, plot=False)
        
            if plot :
            
                if plot_total_polarization :
                    if plot_theta :
                        x, y, yerr = bin_data(ts['TIME'], ts['polarization_2'], ts['polarization_2_err'], median=False, binsize = binsize)
                        axs[ch].plot(ts['TIME'],ts['polarization_2'],"-",color=colors[ch], alpha=0.3, label=bands[ch])
                    else :
                        x, y, yerr = bin_data(ts['TIME'], ts['polarization_1']*100, ts['polarization_1_err']*100, median=False, binsize = binsize)
                        axs[ch].plot(ts['TIME'],ts['polarization_1']*100,"-",color=colors[ch], alpha=0.3, label=bands[ch])
                else :
                    if plot_u :
                        x, y, yerr = bin_data(ts['TIME'], ts['polarization_2']*100, ts['polarization_2_err']*100, median=False, binsize = binsize)
                        axs[ch].plot(ts['TIME'],ts['polarization_2']*100,"-",color=colors[ch], alpha=0.3, label=bands[ch])
                    else :
                        x, y, yerr = bin_data(ts['TIME'], ts['polarization_1']*100, ts['polarization_1_err']*100, median=False, binsize = binsize)
                        axs[ch].plot(ts['TIME'],ts['polarization_1']*100,"-",color=colors[ch], alpha=0.3, label=bands[ch])

                axs[ch].plot(x,y,":",color='k', alpha=0.5)
                                
            lcs.append(ts)
        except :
            lcs.append(None)
            continue
                
        if plot :
            axs[ch].tick_params(axis='x', labelsize=14)
            axs[ch].tick_params(axis='y', labelsize=14)
            axs[ch].minorticks_on()
            axs[ch].tick_params(which='minor', length=3, width=0.7, direction='in',bottom=True, top=True, left=True, right=True)
            axs[ch].tick_params(which='major', length=7, width=1.2, direction='in',bottom=True, top=True, left=True, right=True)
            axs[ch].set_xlabel(r"time (BJD)", fontsize=16)
            
            if plot_total_polarization :
                if plot_theta :
                    axs[ch].set_ylabel(r"$\theta$ (deg)", fontsize=16)
                else :
                    axs[ch].set_ylabel(r"p (%)", fontsize=16)
            else :
                if plot_u :
                    axs[ch].set_ylabel(r"u (%)", fontsize=16)
                else :
                    axs[ch].set_ylabel(r"q (%)", fontsize=16)
            axs[ch].legend(fontsize=16)

    if plot :
        plt.show()
        
    return lcs


def calibrate_astrometry_in_stack_catalog(objprods, use_catalog=False, aperture_radius=10, max_number_of_catalog_sources=1000, output_suffix="_mod.fits", sip_degree=3, niter=1, twirl_find_peak_threshold=1.0, fov_search_factor=1.2, use_twirl_to_compute_wcs=False, search_radius=5.6, plot=False, pol_beam="N") :

    """ Module to re-calibrate astrometry in the stack frames
    
    Parameters
    ----------
    objprods: dict
        data container to store products of analysis

    
    Returns
        catalogs, objprods:
            
    -------
    """
    
    polarimetry = False
    bands = ['g', 'r', 'i', 'z']
    
    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    cat_suffix = ""
    if objprods["instmode"] == "POLAR" :
        catalog_ext = "CATALOG_POL_{}_AP{:03d}".format(pol_beam,aperture_radius)
        cat_suffix = "{}_".format(pol_beam)
        polarimetry = True
             
    objprods["astrometry_calibration"] = [None, None, None, None]
        
    catalogs = []
    for ch in range(4) :
        
        print("CHANNEL: {}".format(ch+1))
        print("STACK image: {}".format(objprods["stack"][ch]))
            
        hdul = fits.open(objprods["stack"][ch])
        img_data, hdr = hdul[0].data, hdul[0].header
        catalog = Table(hdul[catalog_ext].data)
            
        wcs = WCS(hdr, naxis=2)

        pixel_coords = None
        if use_catalog :
            pixel_coords = np.ndarray((len(catalog["X"]), 2))
            for j in range(len(catalog["X"])) :
                pixel_coords[j] = [catalog["X"][j],catalog["Y"][j]]
                #print(j, pixel_coords[j])
        
        #print("###### ORIGINAL WCS ###########")
        #print(repr(wcs.to_header()))
        
        wcs = s4pipelib.astrometry_from_existing_wcs(wcs, img_data, pixel_coords=pixel_coords, sky_coords=None, pixel_scale=0.335, fov_search_factor=fov_search_factor, sparsify_factor=0.01, apply_sparsify_filter=True, max_number_of_catalog_sources=max_number_of_catalog_sources, compute_wcs_tolerance=10, plot_solution=plot, nsources_to_plot=1000, use_twirl_to_compute_wcs=use_twirl_to_compute_wcs, twirl_find_peak_threshold=twirl_find_peak_threshold, sip_degree=sip_degree, use_vizier=False, vizier_catalogs=["UCAC"], vizier_catalog_idx=0)

        #print("###### UPDATED WCS ###########")
        #print(repr(wcs.to_header()))
       
        for iter in range(niter) :
                
            print("iter={} matching catalog sources with Gaia DR2 ...".format(iter))
            catalog_match = match_catalog_sources_with_vizier(objprods["stack"][ch], wcs=wcs, catalog_ext=catalog_ext, search_radius=search_radius, vizier_catalog_name='GAIADR2', plot=plot)
            #catalog_match = match_catalog_sources_with_gaiadr3(objprods["stack"][ch], wcs=wcs, catalog_ext=catalog_ext, search_radius=search_radius, max_G_mag=18, clean_unmatched=False, verbose=False, plot=True)
            #catalog_match.write("gaia_matched_catalog.csv")
            
            matched = catalog_match['GAIADR2 ID'] != 0
            
            if iter == niter - 1 :
                output_catalog_matched = objprods["stack"][ch].replace("stack.fits","{}matched_catalog.csv".format(cat_suffix))
                catalog_match[matched].write(output_catalog_matched,overwrite=True)
            
            pix_coords = []
            matched_sky_coords = []
            for i in range(len(catalog_match['X'][matched])) :
                pix_coords.append([catalog_match['X'][matched][i], catalog_match['Y'][matched][i]])
                matched_sky_coords.append([catalog_match['ra_deg'][matched][i],catalog_match['dec_deg'][matched][i]])
            pix_coords = np.array(pix_coords)
            matched_sky_coords = np.array(matched_sky_coords)

            print("iter={} fitting WCS from matched sources ...".format(iter))
            sky_coords = SkyCoord(matched_sky_coords, unit='deg')
            wcs = fit_wcs_from_points(np.array([catalog_match['X'][matched], catalog_match['Y'][matched]]), sky_coords, proj_point='center', projection='TAN', sip_degree=sip_degree)
            
            print("iter={} calculating astrometric precision ...".format(iter))
            ###### Previous results before fitting wcs ##########
            delta_ra = ((catalog_match['OBS_RA'][matched] - catalog_match['ra_deg'][matched])) * np.cos(catalog_match['dec_deg'][matched]*np.pi/180.)
            delta_dec = (catalog_match['OBS_DEC'][matched] - catalog_match['dec_deg'][matched])
            rms_ra = np.std(delta_ra) * 60 * 60
            rms_dec = np.std(delta_dec) * 60 * 60
            print("PREVIOUS -> RMS RA: {:.5f} arcsec  RMS Dec: {:.5f} arcsec".format(rms_ra,rms_dec))
            ##########################################
        
            ###### New results after fitting wcs ##########
            obs_skycoords = np.array(wcs.pixel_to_world_values(pix_coords))
            deltara, deltadec = np.array([]), np.array([])
            for i in range(len(matched_sky_coords)) :
                deltara = np.append(deltara,np.abs(matched_sky_coords[i][0] - obs_skycoords[i][0])* np.cos(catalog_match['dec_deg'][matched]*np.pi/180.))
                deltadec = np.append(deltadec,np.abs(matched_sky_coords[i][1] - obs_skycoords[i][1]))
            rmsra = np.std(deltara) * 60 * 60
            rmsdec = np.std(deltadec) * 60 * 60
            print("NEW -> RMS RA: {:.5f} arcsec RMS Dec: {:.5f} arcsec".format(rmsra, rmsdec))
            ##########################################
            
            
            
        catalog = get_sci_image_catalog(objprods["stack"][ch], catalog_ext=catalog_ext, wcs=wcs)
        #if plot :
        #    cat_sources_pixcoords = np.array(wcs.world_to_pixel_values(matched_sky_coords))
        #    plt.imshow(img_data, vmin=np.median(img_data), vmax=2 * np.median(img_data), cmap="Greys_r")
        #    _ = photutils.aperture.CircularAperture(cat_sources_pixcoords, r=8.0).plot(color="g")
        #    plt.show()
        objprods["astrometry_calibration"][ch] = deepcopy(wcs)
            
        catalogs.append(catalog)
        
    return catalogs, objprods


def convert_diffmag_to_relflux(mag, emag, normalize=False) :

    flux, eflux = np.array([]), np.array([])
    
    for i in range(len(mag)) :
    
        umag = ufloat(mag[i], emag[i])
        uflux = 10 ** (- 0.4 * umag)
        
        flux = np.append(flux, uflux.nominal_value)
        eflux = np.append(eflux, uflux.std_dev)

    mflux = np.nanmedian(flux)
    
    if normalize :
        flux /= mflux
        eflux /= mflux
        
    return flux, eflux


def get_fluxes_from_lcs(lcs, get_comparisons_independently=True, times=[], fluxes=[], fluxerrs=[], instrument_indexes=[]) :
    
    for ch in range(4) :

        lc = lcs[ch]
        if lc is None :
            continue
            
        if get_comparisons_independently :
        
            ncomps = int( (len(lc.colnames) - 11) / 2)
            
            for i in range(ncomps) :
                diffmag_key = 'diffmag_C{:05d}'.format(i)
                diffmag_err_key = 'diffmag_err_C{:05d}'.format(i)

                diffmag = -np.array(lc[diffmag_key],dtype=float)
                diffmag_err = np.array(lc[diffmag_err_key],dtype=float)
                flux, eflux = convert_diffmag_to_relflux(diffmag, diffmag_err, normalize=True)
            
                times.append(np.array(lc['TIME'],dtype=float))
                fluxes.append(flux)
                fluxerrs.append(eflux)
                instrument_indexes.append(ch+1)
                
        else :
        
            diffmag = -np.array(lc['diffmagsum'],dtype=float)
            diffmag_err = np.array(lc['magsum_err'],dtype=float)
            flux, eflux = convert_diffmag_to_relflux(diffmag, diffmag_err, normalize=True)
            times.append(np.array(lc['TIME'],dtype=float))
            fluxes.append(flux)
            fluxerrs.append(eflux)
            instrument_indexes.append(ch+1)


    return times, fluxes, fluxerrs, instrument_indexes


def get_wppos(tsfile, target=0, comps=[], extname="BEST_APERTURES", verbose=False) :

    wppositions = []
    
    hdul = fits.open(tsfile)
    tbl = Table(hdul[extname].data)

    targettbl = tbl[tbl['SRCINDEX']==target]
    
    try :
        # read comparisons flux data
        for j in range(len(comps)) :
            comptbl = tbl[tbl['SRCINDEX'] == comps[j]]
            comp_wppos = np.array(comptbl['WPPOS'].data)
            wppositions.append(comp_wppos)
            
            #print(j,comps[j],len(comptbl['TIME'].data),len(comp_wppos))
            #plt.plot(comptbl['TIME'].data,comp_wppos)
            
    except :
        if verbose :
            print("WARNING: could not find 'WPPOS' column, exiting ... ")
        pass
    #plt.show()
    return wppositions


def color_match_products(objprods, convolve_images=False, kernel_size=2.0, base_ch=0, reproject_order='bicubic') :

    bands = ['g', 'r', 'i', 'z']
        
    if convolve_images :
        kernel = Gaussian2DKernel(x_stddev=kernel_size)
        
    base_wcs = objprods["astrometry_calibration"][base_ch]
    base_hdul = fits.open(objprods["stack"][base_ch])
    base_hdr = base_hdul[0].header
    base_shape = np.shape(base_hdul[0].data)
    base_wcs_hdr = base_wcs.to_header(relax=True)

    objprods["colormatch_products"] = [None, None, None, None]
        
    for ch in range(4) :
        
        hdul = fits.open(objprods["stack"][ch])
        wcs = objprods["astrometry_calibration"][ch]
        nd = NDData(hdul[0].data, wcs=wcs)
        array, footprint = reproject_interp(nd, output_projection=base_wcs, shape_out=base_shape, order=reproject_order)
            
        if convolve_images :
            array = convolve(array, kernel)
                
        out_filename = objprods["stack"][ch].replace(".fits","_colormatch.fits")
        wcs_hdr = base_wcs.to_header(relax=True)
        out_hdr = hdul[0].header + base_wcs_hdr
        out_hdu = fits.PrimaryHDU(data=array, header=out_hdr)
        out_hdul = fits.HDUList([out_hdu])
        out_hdul.writeto(out_filename,overwrite=True)
            
        objprods["colormatch_products"][ch] = out_filename

    return objprods



def photometric_solution_in_timeseries(objprods, master_catalog, latitude=-22.5344444, longitude=-45.5825, altitude=1864, aperture_radius=12, use_calibrated_magnitudes=False, time_binsize=0.005, use_median=False, verbose=False, plot=False, pol_beam='S+N') :

    catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
    polarimetry = False
    if objprods["instmode"] == "POLAR" :
        if pol_beam == 'S+N' :
            catalog_ext = "CATALOG_PHOT_AP{:03d}".format(aperture_radius)
        else :
            catalog_ext = "CATALOG_POL_{}_AP{:03d}".format(pol_beam,aperture_radius)
        polarimetry = True
        
    # set observatory location
    observatory_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude*u.m)

    bands = ['g', 'r', 'i', 'z']
    colors = ['darkblue','darkgreen','darkorange','darkred']
    
    src_idxs = []
    mags, emags = [], []
    gaia_id = []
    ra, dec, e_ra, e_dec = [], [], [], []
    pmRA, pmDE = [], []
    epoch_yr = []
    
    # first collect source indexes for all sources with measurements in all four bands
    for i in range(len(master_catalog['ra'])) :
    
        has_all_channels = True
        
        srcindex = [None, None, None, None]
        srcmags = [None, None, None, None]
        srcemags = [None, None, None, None]

        for ch in range(4) :
            srcidx_key = 'srcindex_{}'.format(bands[ch])
            srcmag_key = '_{}mag'.format(bands[ch])
            srcemag_key = '_e_{}mag'.format(bands[ch])
            
            if np.isfinite(master_catalog[srcidx_key][i]) and np.isfinite(master_catalog[srcmag_key][i]) and master_catalog[srcmag_key][i] > 1 and np.isfinite(master_catalog[srcemag_key][i]) and master_catalog[srcemag_key][i] != 0 :
            
                srcindex[ch] = master_catalog[srcidx_key][i]
                srcmags[ch] = master_catalog[srcmag_key][i]
                srcemags[ch] = master_catalog[srcemag_key][i]
                
            else :
                has_all_channels = False
                
        if has_all_channels :
            src_idxs.append(srcindex)
            
            gaia_id.append(master_catalog['gaiadr2_id'][i])
            ra.append(master_catalog['ra'][i])
            dec.append(master_catalog['dec'][i])
            e_ra.append(master_catalog['e_ra'][i])
            e_dec.append(master_catalog['e_dec'][i])
            pmRA.append(master_catalog['pmRA'][i])
            pmDE.append(master_catalog['pmDE'][i])
            epoch_yr.append(master_catalog['epoch_yr'][i])

            mags.append(srcmags)
            emags.append(srcemags)


    # set output table
    tbl = Table()
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
    #--------

    a, b = [0, 0, 0, 0], [1, 1, 1, 1]
    catalog_tbl = [None, None, None, None]
    min_time, max_time = 0.,0.
    
    for ch in range(4) :
        if objprods["phot_calibration"][ch] is not None :
            a[ch], b[ch] = objprods["phot_calibration"][ch]
        # open time series fits file
        hdul = fits.open(objprods["lc"][ch])
        # get big table with data
        catalog_tbl[ch] = Table(hdul[catalog_ext].data)
        
        #get min and max time
        min_time = np.nanmin(catalog_tbl[ch]['TIME'].data)
        max_time = np.nanmax(catalog_tbl[ch]['TIME'].data)

    min_npoints_per_bin = 3
    time_bins = np.arange(min_time, max_time, time_binsize)
    nbins = len(time_bins)
   
    catalog_has_row = False
    
    for i in range(len(src_idxs)) :
    
        for ch in range(4) :
        
            # get source index for the current band
            srcindex = src_idxs[i][ch]
            
            # set source coordinates
            source = SkyCoord(ra[i], dec[i], unit=(u.deg, u.deg),frame='icrs', equinox="J2000.0")
            
            # get data for a given source
            src_tbl = catalog_tbl[ch][catalog_tbl[ch]["SRCINDEX"] == srcindex]
        
            # get time data
            time = src_tbl['TIME'].data
            # get photometry data
            obs_mag = src_tbl['MAG'].data
            obs_emag = src_tbl['EMAG'].data
        
            if use_calibrated_magnitudes :
                # Transform from instrumental magnitudes to calibrated magnitudes
                for j in range(len(obs_mag)) :
                    new_mag = (ufloat(obs_mag[j],obs_emag[j]) - a[ch]) / b[ch]
                    obs_mag[j] = new_mag.nominal_value
                    obs_emag[j] = new_mag.std_dev
                        
            if verbose :
                # print source information
                print("i={} srcindex:{} {}-band mag:{}+-{} CH={} Nobs={}".format(i,srcindex,bands[ch],mags[i][ch],emags[i][ch],ch+1,len(time)))
        
            # set times as Time objects
            obstime = Time(time, format='jd', scale='tcb', location=observatory_location)
            # calculate airmasses
            airmass = source.transform_to(AltAz(obstime=obstime, location=observatory_location)).secz
            
            # create bins for each quantity
            bjd_bins = np.full_like(time_bins,np.nan)
            airmass_bins = np.full_like(time_bins,np.nan)
            mag_bins = np.full_like(time_bins,np.nan)
            emag_bins = np.full_like(time_bins,np.nan)

            # create bin masks
            digitized = np.digitize(time, time_bins)
            
            # iterate over each bin to calculate mean quantities
            for j in range(nbins) :
            
                # set index to store data in the final output table
                idx = i*nbins + j
                
                if len(time[digitized == j]) > min_npoints_per_bin :
                
                    if use_median :
                        mag_bins[j] = np.nanmedian(obs_mag[digitized == j])
                        emag_bins[j] = np.nanmedian(np.abs(obs_mag[digitized == j] - mag_bins[j]))  / 0.67449
                    else :
                        mag_bins[j], emag_bins[j] = odd_ratio_mean(obs_mag[digitized == j], obs_emag[digitized == j])
                    
                    bjd_bins[j] = np.nanmean(time[digitized == j])
                    airmass_bins[j] = np.nanmean(airmass[digitized == j])
                    
                # if idx is greater thant master catalog length, create new row and populate it
                if idx >= len(tbl) :
                    tbl.add_row()
                    tbl['gaiadr2_id'][idx] = gaia_id[i]
                    tbl['ra'][idx] = ra[i]
                    tbl['dec'][idx] = dec[i]
                    tbl['e_ra'][idx] = e_ra[i]
                    tbl['e_dec'][idx] = e_dec[i]
                    tbl['pmRA'][idx] = pmRA[i]
                    tbl['pmDE'][idx] = pmDE[i]
                    tbl['epoch_yr'][idx] = epoch_yr[i]
                    tbl['time_bjd'][idx] = bjd_bins[j]
                    tbl['airmass'][idx] = airmass_bins[j]

                tbl['mag_{}'.format(bands[ch])][idx] = mags[i][ch]
                tbl['emag_{}'.format(bands[ch])][idx] =  emags[i][ch]
                tbl['obs_mag_{}'.format(bands[ch])][idx] = mag_bins[j]
                tbl['obs_emag_{}'.format(bands[ch])][idx] = emag_bins[j]

            
            if plot :
                # get median source magnitude
                m_obs_mag = np.nanmedian(obs_mag)
                #plt.plot(airmass, obs_mag-m_obs_mag, '.', alpha=0.3)
                #plt.errorbar(airmass_bins, mag_bins, yerr=emag_bins, fmt='o')
                plt.plot(time, obs_mag-m_obs_mag, '.', alpha=0.3, color=colors[ch], label="{}-mag = {:.3f}".format(bands[ch],mags[i][ch]))
                plt.errorbar(bjd_bins, mag_bins-m_obs_mag, yerr=emag_bins, fmt='o', color=colors[ch])
        
        if plot :
            plt.ylabel(r"$\Delta$mag", fontsize=18)
            plt.xlabel(r"Time [BJD]", fontsize=18)
            plt.legend(fontsize=14)
            plt.show()
        
    return tbl[np.isfinite(tbl['time_bjd'])]

