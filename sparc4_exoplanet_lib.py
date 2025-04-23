"""
    Created on Aug 23 2024

    Description: Library of recipes for the analysis of SPARC4 exoplanet data

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

import sys, os
import numpy as np
from astropy.io import ascii

from uncertainties import ufloat, umath

from scipy import constants
import batman


sparc4_exoplanet_lib_dir = os.path.dirname(__file__)
claret_coeff_path = os.path.join(sparc4_exoplanet_lib_dir, 'data/tess_claret_ldc.dat')

def get_exoplanet(planet_name, exoplanet_catalog=None) :

    loc = {}
    
    # load exoplanet.eu catalog
    if os.path.exists(exoplanet_catalog) :
        exoplanets = ascii.read(exoplanet_catalog)
    else :
        print("WARNING: Could not find exoplanet catalog file: {}".format(exoplanet_catalog))
        return loc

    if planet_name in exoplanets["name"] :
        # select exoplanet
        selected_planet = exoplanets[exoplanets["name"] == planet_name]
    
        for col in selected_planet.colnames :
            if selected_planet[col][0] :
                loc[col] = selected_planet[col][0]
            else :
                loc[col] = None

    else :
        print("WARNING: Could not find exoplanet id={} in exoplanet catalog file: {}".format(planet_name, exoplanet_catalog))

    return loc


def set_exoplanet_variable(exoplanet, variable, default_value=0) :
    
    var = default_value
    if type(var) is float :
        var = ufloat(var,0)

    if exoplanet[variable] :
        var = ufloat(exoplanet[variable],0)
        
    emin = "{}_error_min".format(variable)
    emax = "{}_error_max".format(variable)
    
    if  exoplanet[variable] and exoplanet[emin] :
        var = ufloat(exoplanet[variable],exoplanet[emin])

    if  exoplanet[variable] and exoplanet[emax] :
        var = ufloat(exoplanet[variable],exoplanet[emax])

    if exoplanet[variable] and exoplanet[emin] and  exoplanet[emax] :
        var = ufloat(exoplanet[variable],(exoplanet[emin]+exoplanet[emax])/2)

    return var
    
    
def create_planet_prior_file(exoplanet, output, hasrvdata=False, circular_orbit=True, set_star_density=False, use_impact_parameter=False, planet_index=0, n_planets=1, append=False, all_parameters_fixed=False) :

    teff = set_exoplanet_variable(exoplanet, 'star_teff', default_value=5800.)
    feh = set_exoplanet_variable(exoplanet, 'star_metallicity', default_value=0.)
    ms = set_exoplanet_variable(exoplanet, 'star_mass', default_value=1.)
    rs = set_exoplanet_variable(exoplanet, 'star_radius', default_value=1.)

    rhos = star_density(ms, rs)
    
    per = set_exoplanet_variable(exoplanet, 'orbital_period', default_value=3.)
    a = set_exoplanet_variable(exoplanet, 'semi_major_axis', default_value=semi_major_axis(per, ms, rs, over_rs=True))
    inc = set_exoplanet_variable(exoplanet, 'inclination', default_value=90.)
    tt = set_exoplanet_variable(exoplanet, 'tzero_tr', default_value=0.)

    omega = set_exoplanet_variable(exoplanet, 'omega', default_value=90.)
    ecc = set_exoplanet_variable(exoplanet, 'eccentricity', default_value=0.)
        
    mp = set_exoplanet_variable(exoplanet, 'mass', default_value=1)
    msini = set_exoplanet_variable(exoplanet, 'mass_sini', default_value=mp*umath.sin(inc))
    if not exoplanet['mass'] :
        mp = set_exoplanet_variable(exoplanet, 'mass', default_value=msini)

    rp = set_exoplanet_variable(exoplanet, 'radius', default_value=1.)
    
    rsun, rjup = 696.34e6, 69.911e6
    rp_over_rs = (rp * rjup) / (rs * rsun)

    k = set_exoplanet_variable(exoplanet, 'k', default_value=rv_semi_amplitude(per, inc, ecc, ms, mp)[0])
    b = umath.cos(inc * np.pi / 180) * a

    u0, u1 = get_claret_ld_coeffs(teff.nominal_value)
    u0 = ufloat(u0,0)
    u1 = ufloat(u1,0)
    
    if append :
        outfile = open(output,"a+")
    else :
        outfile = open(output,"w+")
      
    if not append :
        outfile.write("# Parameter_ID\tPrior_Type\tValues\n")
        if all_parameters_fixed :
            outfile.write("teff\tFIXED\t{:.0f}\n".format(teff.nominal_value))
            outfile.write("ms\tFIXED\t{:.3f}\n".format(ms.nominal_value))
            outfile.write("rs\tFIXED\t{:.3f}\n".format(rs.nominal_value))
            outfile.write("feh\tFIXED\t{:.3f}\n".format(feh.nominal_value))
            if set_star_density :
                outfile.write("rhos\FIXED\t{:.3f}\n".format(rhos.nominal_value))
        else :
            outfile.write("teff\tNormal\t{:.0f},{:.0f}\n".format(teff.nominal_value, teff.std_dev))
            outfile.write("ms\tNormal\t{:.3f},{:.3f}\n".format(ms.nominal_value, ms.std_dev))
            outfile.write("rs\tNormal\t{:.3f},{:.3f}\n".format(rs.nominal_value, rs.std_dev))
            outfile.write("feh\tFIXED\t{:.3f},{:.3f}\n".format(feh.nominal_value, feh.std_dev))
            if set_star_density :
                outfile.write("rhos\tNormal\t{:.3f},{:.3f}\n".format(rhos.nominal_value, rhos.std_dev))
                
        outfile.write("n_planets\tFIXED\t{}\n".format(n_planets))
        
    outfile.write("transit_{:03d}\tFIXED\t1\n".format(planet_index))

    if hasrvdata and not all_parameters_fixed :
        outfile.write("k_{:03d}\tUniform\t-1000,1000.,{:.3f}\n".format(planet_index,k.nominal_value))
    elif not hasrvdata and not all_parameters_fixed :
        outfile.write("k_{:03d}\tNormal\t{:.3f}\t{:.3f}\n".format(planet_index,k.nominal_value,k.std_dev))
    else :
        outfile.write("k_{:03d}\tFIXED\t{:.3f}\n".format(planet_index,k.nominal_value))

    outfile.write("rvsys_{:03d}\tFIXED\t0.\n".format(planet_index))
    outfile.write("trend_{:03d}\tFIXED\t0.\n".format(planet_index))
    outfile.write("quadtrend_{:03d}\tFIXED\t0.\n".format(planet_index))

    if all_parameters_fixed :
        outfile.write("tc_{:03d}\tFIXED\t{:.8f}\n".format(planet_index,tt.nominal_value))
        outfile.write("per_{:03d}\tFIXED\t{:.8f}\n".format(planet_index,per.nominal_value))
        if not set_star_density :
            outfile.write("a_{:03d}\tFIXED\t{:.2f}\n".format(planet_index,a.nominal_value))
        outfile.write("rp_{:03d}\tFIXED\t{:.5f}\n".format(planet_index,rp.nominal_value))
        if use_impact_parameter :
            outfile.write("b_{:03d}\tFIXED\t{:.2f}\n".format(planet_index,b.nominal_value))
        else :
            outfile.write("inc_{:03d}\tFIXED\t{:.2f}\n".format(planet_index,inc.nominal_value))
        outfile.write("u0_{:03d}\tFIXED\t{:.4f}\n".format(planet_index,u0.nominal_value))
        outfile.write("u1_{:03d}\tFIXED\t{:.4f}\n".format(planet_index,u1.nominal_value))
    else :
        outfile.write("tc_{:03d}\tUniform\t{:.8f},{:.8f},{:.8f}\n".format(planet_index,tt.nominal_value-0.1*per.nominal_value,tt.nominal_value+0.1*per.nominal_value,tt.nominal_value))
        outfile.write("per_{:03d}\tUniform\t{:.8f},{:.8f},{:.8f}\n".format(planet_index,per.nominal_value*0.8,per.nominal_value*1.2,per.nominal_value))
        if not set_star_density :
            outfile.write("a_{:03d}\tUniform\t{:.2f},{:.2f},{:.2f}\n".format(planet_index,a.nominal_value*0.5,a.nominal_value*2.0,a.nominal_value))
        outfile.write("rp_{:03d}\tUniform\t{:.5f},{:.5f},{:.5f}\n".format(planet_index,rp_over_rs.nominal_value*0.8,rp_over_rs.nominal_value*1.2,rp_over_rs.nominal_value))
        if use_impact_parameter :
            outfile.write("b_{:03d}\tUniform\t0,1,{:.5f}\n".format(planet_index,b.nominal_value))
        else :
            outfile.write("inc_{:03d}\tUniform\t{:.2f},90,{:.2f}\n".format(planet_index,inc.nominal_value*0.8,inc.nominal_value))
        outfile.write("u0_{:03d}\tUniform\t0.,3.,{:.4f}\n".format(planet_index,u0.nominal_value))
        outfile.write("u1_{:03d}\tUniform\t0.,3.,{:.4f}\n".format(planet_index,u1.nominal_value))
    
    if circular_orbit or all_parameters_fixed :
        outfile.write("ecc_{:03d}\tFIXED\t{:.5f}\n".format(planet_index,ecc.nominal_value))
        outfile.write("w_{:03d}\tFIXED\t{:.5f}\n".format(planet_index,ecc.nominal_value))
    else :
        outfile.write("ecc_{:03d}\tUniform\t0,1,{:.5f}\n".format(planet_index,ecc.nominal_value))
        outfile.write("w_{:03d}\tUniform\t0.,360.,{:.3f}\n".format(planet_index,w.nominal_value))

    outfile.close()



def batman_transit_model(time, planet_params, planet_index=0) :

    """
        Function for computing transit models for the set of 8 free paramters
        x - time array
        """
    params = batman.TransitParams()
    
    params.per = planet_params['per_{0:03d}'.format(planet_index)]
    params.t0 = planet_params['tc_{0:03d}'.format(planet_index)]
    
    if any("a_" in key for key in planet_params.keys()) :
        params.a = planet_params['a_{0:03d}'.format(planet_index)]
    elif any("rhos" in key for key in planet_params.keys()) and not any("a_" in key for key in planet_params.keys()) :
        rhos = planet_params['rhos']
        params.a = semi_major_axis_from_star_density(params.per, rhos)
        
    if any("inc_" in key for key in planet_params.keys()) :
        params.inc = planet_params['inc_{0:03d}'.format(planet_index)]
    elif any("b_" in key for key in planet_params.keys()) and not any("inc_" in key for key in planet_params.keys()) :
        b = planet_params['b_{0:03d}'.format(planet_index)]
        params.inc = np.arccos(b/params.a)*180/np.pi
    
    if params.inc > 90 :
        params.inc = 180-params.inc
   
    params.ecc = planet_params['ecc_{0:03d}'.format(planet_index)]
    params.w = planet_params['w_{0:03d}'.format(planet_index)]
    params.rp = planet_params['rp_{0:03d}'.format(planet_index)]
    u0 = planet_params['u0_{0:03d}'.format(planet_index)]
    u1 = planet_params['u1_{0:03d}'.format(planet_index)]
    
    if u0 < 0 :
        u0 = np.abs(u0)

    if u1 < 0 :
        u1 = np.abs(u1)

    if params.ecc < 0 :
        params.ecc = np.abs(params.ecc)

    params.u = [u0,u1]
    params.limb_dark = "quadratic"       #limb darkening model

    #print(params.per,params.t0,params.a,params.inc,params.ecc,params.w,params.rp,params.u,params.limb_dark)

    m = batman.TransitModel(params, time)    #initializes model
        
    flux_m = m.light_curve(params)          #calculates light curve

    return np.array(flux_m)


def batman_model(time, per, t0, a, inc, rp, u0, u1=0., ecc=0., w=90.) :
    
    """
        Function for computing transit models for the set of 8 free paramters
        x - time array
        """
    params = batman.TransitParams()
    
    params.per = per
    params.t0 = t0
    params.inc = inc
    params.a = a
    params.ecc = ecc
    params.w = w
    params.rp = rp
    params.u = [u0,u1]
    params.limb_dark = "quadratic"       #limb darkening model
    
    m = batman.TransitModel(params, time)    #initializes model
    
    flux_m = m.light_curve(params)          #calculates light curve
    
    return np.array(flux_m)


def rv_model(t, per, tp, ecc, om, k):
    """RV Drive
    Args:
        t (array of floats): times of observations (JD)
        per (float): orbital period (days)
        tp (float): time of periastron (JD)
        ecc (float): eccentricity
        om (float): argument of periatron (degree)s
        k (float): radial velocity semi-amplitude (m/s)
        
    Returns:
        rv: (array of floats): radial velocity model
    """

    omega = np.pi * om / 180.
    # Performance boost for circular orbits
    if ecc == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + omega)

    if per < 0:
        per = 1e-4
    if ecc < 0:
        ecc = 0
    if ecc > 0.99:
        ecc = 0.99

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    nu = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(nu + omega) + ecc * np.cos(omega))

    return rv


def kepler(Marr, eccarr):
    """Solve Kepler's Equation
    Args:
        Marr (array): input Mean anomaly
        eccarr (array): eccentricity
    Returns:
        array: eccentric anomaly
    """

    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1

        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(convd is True)

    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]


def timetrans_to_timeperi(tc, per, ecc, om):
    """
    Convert Time of Transit to Time of Periastron Passage
    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (degree)
    Returns:
        float: time of periastron passage
    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass
    omega = om * np.pi / 180.
    f = np.pi/2 - omega
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron

    return tp


def timeperi_to_timetrans(tp, per, ecc, om, secondary=False) :
    """
    Convert Time of Periastron to Time of Transit
    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead
    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)
    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass

    omega = om * np.pi / 180.
    
    if secondary:
        f = 3*np.pi/2 - omega                                       # true anomaly during secondary eclipse
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        f = np.pi/2 - omega                                         # true anomaly during transit
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

    tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))         # time of conjunction

    return tc


def true_anomaly(t, tp, per, ecc):
    """
    Calculate the true anomaly for a given time, period, eccentricity.
    Args:
        t (array): array of times in JD
        tp (float): time of periastron, same units as t
        per (float): orbital period in days
        ecc (float): eccentricity
    Returns:
        array: true anomoly at each time
    """

    # f in Murray and Dermott p. 27
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + ecc
    e1 = kepler(m, eccarr)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))

    return nu

def star_density(mstar, rstar) :
    """
        Calculate the star density
        Args:
        mstar (float): stellar mass in Msun
        rstar (float): stellar radius in Rsun
        Returns:
        rhos (float): stellar mean density in units of g/cm^3
        """
    msun = 1.989e+33 # g
    rsun = 6.9634e10 # cm
    ms = mstar * msun
    rs = rstar * rsun
    rhos = (3 * ms) / (4 * np.pi * rs*rs*rs)
    return rhos

def semi_major_axis_from_star_density(per, star_density) :
    """
        Calculate the semi-major axis
        Args:
        per (float): orbital period in days
        star_density (float): in units of kg/m^3
        Returns:
        a (float): semi-major axis in units of star radius
        """
        
    G = constants.G * 1e3 # CGS
    d2s = 24.*60.*60.
    per_s = d2s*per # convert period from day to second
    a_over_Rs = ((per_s * per_s * G * star_density)/(3. * np.pi ))**(1/3)
    
    return a_over_Rs
 

def semi_major_axis(per, mstar, rstar, over_rs=False) :
    """
        Calculate the semi-major axis
        Args:
        per (float): orbital period in days
        mstar (float): stellar mass in Msun
        rstar (float): stellar radius in Rsun
        over_rs (bool): switch to output units in relative units to star radius
        Returns:
        a (float): semi-major axis in units of au
        """
        
    G = constants.G
    msun = 1.989e+30
    rsun = 696.34e6
    ms = mstar * msun
    rs = rstar * rsun
    d2s = 24.*60.*60.
    
    a = (((d2s*per * d2s*per * G * ms)/(4. * np.pi * np.pi))**(1/3))
    
    if over_rs :
        return a / rs
    else :
        au_m = 1.496e+11
        return a / au_m


def rv_semi_amplitude(per, inc, ecc, mstar, mp) :
    """
        Calculate the radial velocity semi-amplitude
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        mstar (float): stellar mass (Msun)
        mp (float): planet mass (Mjup)
        Returns:
        ks,kp ([float,float]): ks : star radial velocity semi-amplitude
                               kp : planet radial velocity semi-amplitude
        """

    # mpsini and mp in jupiter mass
    # mstar in solar mass
    
    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = per * 24. * 60. * 60. # in s
    
    mjup = 1.898e27 # mass of Jupiter in Kg
    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    mp_kg = mp * mjup
    
    inc_rad = inc * np.pi/180. # inclination in radians
    
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p2 = umath.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
    p3 = 1./umath.sqrt(1 - ecc*ecc)
    
    ks = mp_kg * p1*p2*p3
    kp = mstar_kg * p1*p2*p3
    
    # return semi-amplitudes in km/s
    return ks, kp


def planet_mass(per, inc, ecc, mstar, k, units='') :
    """
        Calculate the planet mass
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        mstar (float): stellar mass (Msun)
        k (float): radial velocity semi-amplitude (km/s)
        units (string) : define the output units for the planet mass
        supported units: 'mjup', 'mnep', 'mearth', 'msun', or in kg by default
        Returns:
        mp (float): planet mass
        """

    G = constants.G # constant of gravitation in m^3 kg^-1 s^-2
    
    per_s = per * 24. * 60. * 60. # in s

    msun = 1.989e30 # mass of the Sun in Kg
    
    mstar_kg = mstar*msun
    inc_rad = inc * np.pi/180. # inclination in radians
    p1 = (2. * np.pi * G / per_s)**(1/3)
    p3 = 1./np.sqrt(1 - ecc*ecc)

    mp_kg = 0.
    for i in range(10) :
        p2 = np.sin(inc_rad) / (mstar_kg + mp_kg)**(2/3)
        mp_kg = k / (p1*p2*p3)
    
    if units == 'mjup':
        mjup = 1.898e27 # mass of Jupiter in Kg
        return mp_kg/mjup
    elif units == 'mnep':
        mnep = 1.024e26
        return mp_kg/mnep
    elif units == 'mearth' :
        mearth = 5.972e24
        return mp_kg/mearth
    elif units == 'msun' :
        return mp_kg/msun
    else :
        return mp_kg


def transit_duration(per, inc, ecc, om, rp, mstar, rstar):
    """
        Calculate the transit duration
        Args:
        per (float): orbital period (day)
        inc (float): orbital inclination (degree)
        ecc (float): eccentricity
        om (float): argument of periastron (degree)
        rp (float): planet radius (Rjup)
        mstar (float): stellar mass (Msun)
        rstar (float): stellar radius (Rsun)
        Returns:
        tdur (float): transit duration (days)
        """
    rsun = 696.34e6
    rs = rstar * rsun
    rjup = 69.911e6
    rp_over_rs = rp * rjup / rs
    
    sma_over_rs = semi_major_axis(per, mstar, rstar, over_rs=True)

    ww = om * np.pi / 180
    ii = inc * np.pi / 180
    ee = ecc
    aa = sma_over_rs
    ro_pt = (1 - ee ** 2) / (1 + ee * np.sin(ww))
    b_pt = aa * ro_pt * np.cos(ii)
    if b_pt > 1:
        b_pt = 0.5
    s_ps = 1.0 + rp_over_rs
    df = np.arcsin(np.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    tdur = (per * (ro_pt ** 2)) / (np.pi * np.sqrt(1 - ee ** 2)) * df
    
    return tdur



def aflare(t, p):
    """
    This is the Analytic Flare Model from the flare-morphology paper.
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Note: this model assumes the flux before the flare is zero centered
    Note: many sub-flares can be modeled by this method by changing the
    number of parameters in "p". As a result, this routine may not work
    for fitting with methods like scipy.optimize.curve_fit, which require
    a fixed number of free parameters. Instead, for fitting a single peak
    use the aflare1 method.
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    p : 1-d array
        p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    Nflare = int( np.floor( (len(p)/3.0) ) )
    #print(Nflare, p)
    flare = np.zeros_like(t)
    # compute the flare model for each flare
    for i in range(Nflare):
        outm = np.piecewise(t, [(t<= p[0+i*3]) * (t-p[0+i*3])/p[1+i*3] > -1.,
                                (t > p[0+i*3])],
                            [lambda x: (_fr[0]+                             # 0th order
                                        _fr[1]*((x-p[0+i*3])/p[1+i*3])+     # 1st order
                                        _fr[2]*((x-p[0+i*3])/p[1+i*3])**2.+  # 2nd order
                                        _fr[3]*((x-p[0+i*3])/p[1+i*3])**3.+  # 3rd order
                                        _fr[4]*((x-p[0+i*3])/p[1+i*3])**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[3] ))]
                            ) * p[2+i*3] # amplitude
        flare = flare + outm

    return flare


def calib_model(n, i, params, time) :
    #polynomial model
    ncoefs = int(len(params) / n)
    coefs = []
    for c in range(int(ncoefs)):
        coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
        coefs.append(params[coeff_id])
    #p = np.poly1d(np.flip(coefs))
    p = np.poly1d(coefs)
    out_model = p(time)
    return out_model


def flares_model(flare_params, flare_tags, index, time) :
    n_flares = int(len(flare_params) / 3)
    pflares = []
    for i in range(n_flares) :
        if flare_tags[i] == index :
            tc_id = 'tc{0:04d}'.format(i)
            fwhm_id = 'fwhm{0:04d}'.format(i)
            amp_id = 'amp{0:04d}'.format(i)
            pflares.append(flare_params[tc_id])
            pflares.append(flare_params[fwhm_id])
            pflares.append(flare_params[amp_id])

    flare_model = aflare(time, pflares)
    return flare_model


def Transmission_Spectroscopy_Metric(rp, mp, Rs, Teq, mj, scale_factor=1.0) :
    """
        Description: Transmission Spectroscopy Metric (TSM) of Kempton et al. (2018)
        https://iopscience.iop.org/article/10.1088/1538-3873/aadf6f/pdf
        
        Parameters:
        rp: planet radius in Earth radii
        mp: planet mass in Earth masses
        Rs: stellar radius in solar radii
        Teq: equilibrium temperature in K
        mj: apparent magnitude in J band
        scale_factor : a normalization factor
    """
    
    tsm = scale_factor * ((rp ** 3) * Teq * 10**(-mj/5) ) / (mp * Rs**2)

    return tsm



def get_claret_ld_coeffs(teff, logg=5.0, zsun=0, method="LSM") :
    """
        Description: Function to retrieve quadratic limb darkening
        coefficients from Claret (2018)
        
        https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract
        
        teff: (float), star effective temperature in K
        logg: (float), logarithm of star surface gravity
        zsun: (float), star metallicity
        
        return (float,float) tuple with linear and quadratic coefficients
    """


    lddata = ascii.read(claret_coeff_path)
    
    rounded_logg = round(logg * 2.) / 2.
    
    keep = (lddata['col1'] == rounded_logg) & (lddata['col3'] == zsun)
    
    dtmin = 1e20
    a, b = np.nan, np.nan
    
    if len(lddata['col1'][keep]) == 0 :
        print("ERROR: select 0.0 < logg < 5.0 and -5 < zsun < 1")
        return a, b
    
    for i in range(len(lddata['col2'][keep])) :
        dt = np.abs(lddata['col2'][keep][i]-teff)
        if dt < dtmin :
            dtmin = dt
            if method == "LSM" :
                a = lddata['col5'][keep][i]
                b = lddata['col6'][keep][i]
            elif method == "PCM" :
                a = lddata['col7'][keep][i]
                b = lddata['col8'][keep][i]
            else :
                print("ERROR: invalid limb-darkening method. Select LSM or PCM, exiting...")
                exit()
    return a,b
