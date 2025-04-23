# -*- coding: iso-8859-1 -*-
"""
    Created on November 3 2020
    
    Description: Priors library
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
from copy import deepcopy

def get_quantiles(dist, alpha = 0.68, method = 'median'):
    """
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error%
        credibility around it. This assumes you give a non-ordered
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

class normal_parameter:
    """
      Description
      -----------
      This class defines a parameter object which has a normal prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
      
    def __init__(self,prior_hypp, pos = False):
        self.value = prior_hypp[0]
        self.value_u = prior_hypp[1]
        self.value_l = prior_hypp[1]
        self.prior_hypp = prior_hypp
        self.posterior = []
        self.is_positive = pos

    def get_ln_prior(self):
        return np.log(1./np.sqrt(2.*np.pi*(self.prior_hypp[1]**2)))-\
                 0.5*(((self.prior_hypp[0]-self.value)**2/(self.prior_hypp[1]**2)))

    def set_value(self,new_val):
        self.value = new_val

    def set_posterior(self,posterior_chain):
        self.posterior = posterior_chain
        param, param_u, param_l = get_quantiles(posterior_chain)
        self.value = param
        self.value_u = param_u
        self.value_l = param_l
      
    def check_value(self, x):
        return ((self.is_positive and x < 0) == False)

class uniform_parameter :
    """
      Description
      -----------
      This class defines a parameter object which has a uniform prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
    def __init__(self,prior_hypp):
        if len(prior_hypp) == 3 :
            self.value = prior_hypp[2]
        else :
            self.value = (prior_hypp[0]+prior_hypp[1])/2.

        self.value_l = prior_hypp[0]
        self.value_u = prior_hypp[1]
        self.prior_hypp = prior_hypp
        self.posterior = []

    def get_ln_prior(self):
        return np.log(1./(self.prior_hypp[1]-self.prior_hypp[0]))

    def check_value(self,x):
        if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
            return True
        else:
            return False
 
    def set_value(self,new_val):
        self.value = new_val

    def set_posterior(self,posterior_chain):
        self.posterior = posterior_chain
        param, param_u, param_l = get_quantiles(posterior_chain)
        self.value = param
        self.value_u = param_u
        self.value_l = param_l

class jeffreys_parameter:
    """
      Description
      -----------
      This class defines a parameter object which has a Jeffreys prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
    def __init__(self,prior_hypp):
        if len(prior_hypp) == 3 :
            self.value = prior_hypp[2]
        else :
            self.value = np.sqrt(prior_hypp[0]*prior_hypp[1])

        self.value_u = 0.0
        self.value_l = 0.0
        self.prior_hypp = prior_hypp
        self.posterior = []

    def get_ln_prior(self):
        return np.log(1.0) - np.log(self.value*np.log(self.prior_hypp[1]/self.prior_hypp[0]))

    def check_value(self,x):
        if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
            return True
        else:
            return False

    def set_value(self,new_val):
        self.value = new_val

    def set_posterior(self,posterior_chain):
        self.posterior = posterior_chain
        param, param_u, param_l = get_quantiles(posterior_chain)
        self.value = param
        self.value_u = param_u
        self.value_l = param_l

class constant_parameter:
    """
      Description
      -----------
      This class defines a parameter object which has a constant value. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
    def __init__(self,val):
        self.value = val


def generate_parameter(values):
    """
      Description
      -----------
      Function to initalize main prior dictionary with the characteristics of a given distribution for each parameter.
      """
    out_dict = {}
    out_dict['type'] = values[1]
    if values[1] == 'Normal':
        out_dict['object'] = normal_parameter(np.array(values[2].split(',')).astype('float64'))
    elif values[1] == 'Normal_positive':
        out_dict['object'] = normal_parameter(np.array(values[2].split(',')).astype('float64'), True)
    elif values[1] == 'Uniform':
        out_dict['object'] = uniform_parameter(np.array(values[2].split(',')).astype('float64'))
    elif values[1] == 'Jeffreys':
        out_dict['object'] = jeffreys_parameter(np.array(values[2].split(',')).astype('float64'))
    elif values[1] == 'FIXED':
        out_dict['object'] = constant_parameter(np.array(values[2].split(',')).astype('float64')[0])
    if len(values)>=5:
        out_dict['object'].set_value(np.float(values[3]))
    return out_dict


def read_priors(filename, calibration=False, rvcalibration=False, flares=False):
    """
      Description
      -----------
      Function to read priors from file.
      """

    # Open the file containing the priors:
    f = open(filename)
    # Generate dictionary that will save the data on the priors:
    priors = {}
    n_params = 0
    while True:
        line = f.readline()
        if line == '':
            break
        elif line[0] != '#':
            # Extract values from text file: [0]: parameter name,
            #                                [1]: prior type,
            #                                [2]: hyperparameters,
            #                                [3]: starting value (optional)
            values = line.split()
            priors[values[0]] = generate_parameter(values)
            errors = np.array(values[2].split(',')).astype('float64')
            error_key = "{0}_err".format(values[0])
            pdf_key = "{0}_pdf".format(values[0])
            priors[pdf_key] = values[1]
            priors[error_key] = errors
            n_params += 1

    f.close()

    if calibration :
        n_coefs = n_params
        ndatasets = n_coefs / float(priors['orderOfPolynomial']['object'].value)
        priors["ndatasets"] = generate_parameter(['ndatasets','FIXED',str(ndatasets)])
        baseorder = priors['orderOfPolynomial']['object'].value

    if rvcalibration :
        n_rvdatasets = n_params
        priors["n_rvdatasets"] = generate_parameter(['n_rvdatasets','FIXED',str(n_rvdatasets)])

    if flares :
        # assuming each flare has 3 free parameters
        n_flares = n_params / 3
        priors["n_flares"] = int(n_flares)

    return priors


def read_flares_params(flare_priors_dict, output_theta_params=False) :
    
    n_flares = flare_priors_dict["n_flares"]
    
    param_ids = ['tc', 'amp', 'fwhm']
    
    tc, amp, fwhm = [], [], []
    
    flare_params = {}

    for i in range(n_flares) :
        tc_id = 'tc{0:04d}'.format(i)
        amp_id = 'amp{0:04d}'.format(i)
        fwhm_id = 'fwhm{0:04d}'.format(i)
        flare_params[tc_id] = flare_priors_dict[tc_id]['object'].value
        flare_params[amp_id] = flare_priors_dict[amp_id]['object'].value
        flare_params[fwhm_id] = flare_priors_dict[fwhm_id]['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in flare_params.keys():
            param = flare_priors_dict[key]
            if param['type'] != 'FIXED':
                theta.append(flare_params[key])
                labels.append(key)
        return theta, labels
    else :
        return flare_params


def read_calib_params(calib_priors_dict, output_theta_params = False):

    n_coefs = len(calib_priors_dict) - 1.0
    order_of_polynomial = calib_priors_dict['orderOfPolynomial']['object'].value
    ndatasets = calib_priors_dict['ndatasets']['object'].value

    calib_params = {}
    
    for i in range(int(ndatasets)):
        for c in range(int(order_of_polynomial)):
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
            calib_params[coeff_id] = calib_priors_dict[coeff_id]['object'].value

    if output_theta_params :
        theta, labels = [], []
        for key in calib_params.keys():
            param = calib_priors_dict[key]
            if param['type'] != 'FIXED':
                theta.append(calib_params[key])
                labels.append(key)
        return theta, labels
    else :
        return calib_params


def read_rvcalib_params(rvcalib_priors_dict, output_theta_params = False):
    
    n_rvdatasets = rvcalib_priors_dict['n_rvdatasets']['object'].value
    
    rvcalib_params = {}
    
    for i in range(int(n_rvdatasets)):
        coeff_id = 'rv_d{0:02d}'.format(i)
        rvcalib_params[coeff_id] = rvcalib_priors_dict[coeff_id]['object'].value
        
        jitter_id = 'jitter_d{0:02d}'.format(i)
        rvcalib_params[jitter_id] = rvcalib_priors_dict[jitter_id]['object'].value

    if output_theta_params :
        theta, labels = [], []
        for key in rvcalib_params.keys():
            param = rvcalib_priors_dict[key]
            if param['type'] != 'FIXED':
                theta.append(rvcalib_params[key])
                labels.append(key)
        return theta, labels
    else :
        return rvcalib_params


#intialize calib parameters
def init_rvcalib_priors(ndim=1, coefs=None, jitters=None) :
    """
        ndim : number of polynomials, typically the number of different datasets
        coefs: array of coefficients n=dim
        """
    priors = {}
    
    nds_dict = {}
    nds_dict['type'] = "FIXED"
    nds_dict['object'] = constant_parameter(ndim)
    priors['n_rvdatasets'] = nds_dict

    for i in range(int(ndim)):
        coeff_id = 'rv_d{0:02d}'.format(i)
        out_dict = {}
        out_dict['type'] = "Uniform"
        l_value = -1e20
        u_value = +1e20
        out_dict['object'] = uniform_parameter(np.array([l_value, u_value]))
        if coefs != None :
            out_dict['object'].set_value(np.float(coefs[i][c]))
        priors[coeff_id] = out_dict
        
        jitter_id = 'jitter_d{0:02d}'.format(i)
        out_jitter_dict = {}
        out_jitter_dict['type'] = "FIXED"
        out_jitter_dict['object'] = constant_parameter(0.)
        if jitters != None :
            out_jitter_dict['object'].set_value(np.float(jitters[i]))
        priors[jitter_id] = out_jitter_dict

    return priors


#intialize calib parameters
def init_calib_priors(ndim=1, order=1, coefs=None) :
    """
        ndim : number of polynomials, typically the number of different datasets
        order:  number of coefficients in the polynomial
        coefs: array of arrays of coefficients.
        
        e.g.: if ndim=2, then coefs=[[3,2],[1,3]]
        """
    priors = {}

    order_dict = {}
    order_dict['type'] = "FIXED"
    order_dict['object'] = constant_parameter(order)
    priors['orderOfPolynomial'] = order_dict
    
    nds_dict = {}
    nds_dict['type'] = "FIXED"
    nds_dict['object'] = constant_parameter(ndim)
    priors['ndatasets'] = nds_dict
    
    for i in range(int(ndim)):
        for c in range(order):
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
            out_dict = {}
            out_dict['type'] = "Uniform"
            l_value = -1e20
            u_value = +1e20
            
            out_dict['object'] = uniform_parameter(np.array([l_value, u_value]))
            
            if coefs != None :
                out_dict['object'].set_value(np.float(coefs[i][c]))

            priors[coeff_id] = out_dict

    return priors


def get_theta_from_priors(planet_priors, calib_priors, flare_priors={}, instrum_priors=None, rvcalib_priors=None) :
    
    theta_planet, planet_labels  = [], []
    for i in range(len(planet_priors)) :
        theta_pl, pl_labels = read_exoplanet_params(planet_priors[i], planet_index=i, output_theta_params=True)
        theta_planet = np.concatenate((theta_planet, theta_pl), axis=0)
        planet_labels = planet_labels + pl_labels
    
    theta_calib, calib_labels = read_calib_params(calib_priors, output_theta_params = True)

    if instrum_priors is not None :
        theta_instrum, instrum_labels = read_instrument_params(instrum_priors, output_theta_params=True)
        calib_labels += instrum_labels
        theta_calib = np.concatenate((theta_calib, theta_instrum), axis=0)

    if rvcalib_priors is not None:
        theta_rvcalib, rvcalib_labels = read_rvcalib_params(rvcalib_priors, output_theta_params=True)
        calib_labels += rvcalib_labels
        theta_calib = np.concatenate((theta_calib, theta_rvcalib), axis=0)

    if len(flare_priors) :
        theta_flare, flare_labels = read_flares_params(flare_priors, output_theta_params = True)
        theta_planet_calib = np.concatenate((theta_planet, theta_calib), axis=0)
        theta = np.concatenate((theta_planet_calib, theta_flare), axis=0)
        labels = planet_labels + calib_labels + flare_labels
    else :
        theta = np.concatenate((theta_planet, theta_calib), axis=0)
        labels = planet_labels + calib_labels

    theta_priors = {}
    for i in range(len(planet_priors)) :
        for key in planet_priors[i].keys() :
            if key in labels :
                theta_priors[key] = planet_priors[i][key]
    for key in calib_priors.keys() :
        if key in labels :
            theta_priors[key] = calib_priors[key]
            
    if instrum_priors is not None :
        for key in instrum_priors.keys() :
            if key in labels :
                theta_priors[key] = instrum_priors[key]

    if rvcalib_priors :
        for key in rvcalib_priors.keys() :
            if key in labels :
                theta_priors[key] = rvcalib_priors[key]

    if len(flare_priors) :
        for key in flare_priors.keys() :
            if key in labels :
                theta_priors[key] = flare_priors[key]

    return theta, labels, theta_priors


def save_posterior(output, params, theta_fit, theta_labels, theta_err, calib=False, ncoeff=1, ref_time=0.) :

    outfile = open(output,"w")
    outfile.write("# Parameter_ID\tPrior_Type\tValues\n")
    if calib :
        outfile.write("orderOfPolynomial\tFIXED\t{0}\n".format(ncoeff))

    for key in params.keys() :
        if key in theta_labels :
            idx = theta_labels.index(key)
            error = (theta_err[idx][0] + theta_err[idx][1]) / 2.
            if key == 'tau' :
                outfile.write("{0}\tNormal\t{1:.10f},{2:.10f}\n".format(key, theta_fit[idx]+ref_time, error))
            else :
                outfile.write("{0}\tNormal\t{1:.10f},{2:.10f}\n".format(key, theta_fit[idx], error))
        else :
            if ('_err' not in key) and ('_pdf' not in key) :
                if key == 'tau' :
                    outfile.write("{0}\tFIXED\t{1}\n".format(key,params[key]+ref_time))
                else :
                    outfile.write("{0}\tFIXED\t{1}\n".format(key,params[key]))

    outfile.close()


def get_theta_from_transit_rv_priors(planet_priors, calib_priors, flare_priors={}, instrum_priors=None, rvcalib_priors=None) :
    
    theta_planet, planet_labels = read_exoplanet_transit_rv_params(planet_priors, output_theta_params=True)
    
    theta_calib, calib_labels = read_calib_params(calib_priors, output_theta_params = True)

    if instrum_priors is not None:
        theta_instrum, instrum_labels = read_instrument_params(instrum_priors, output_theta_params=True)
        calib_labels += instrum_labels
        theta_calib = np.concatenate((theta_calib, theta_instrum), axis=0)

    if rvcalib_priors is not None:
        theta_rvcalib, rvcalib_labels = read_rvcalib_params(rvcalib_priors, output_theta_params=True)
        calib_labels += rvcalib_labels
        theta_calib = np.concatenate((theta_calib, theta_rvcalib), axis=0)
        
    if len(flare_priors) :
        theta_flare, flare_labels = read_flares_params(flare_priors, output_theta_params = True)
        theta_planet_calib = np.concatenate((theta_planet, theta_calib), axis=0)
        theta = np.concatenate((theta_planet_calib, theta_flare), axis=0)
        labels = planet_labels + calib_labels + flare_labels
    else :
        theta = np.concatenate((theta_planet, theta_calib), axis=0)
        labels = planet_labels + calib_labels

    theta_priors = {}
    for key in planet_priors.keys() :
        if key in labels :
            theta_priors[key] = planet_priors[key]
            
    for key in calib_priors.keys() :
        if key in labels :
            theta_priors[key] = calib_priors[key]

    if instrum_priors is not None :
        for key in instrum_priors.keys() :
            if key in labels :
                theta_priors[key] = instrum_priors[key]

    if rvcalib_priors :
        for key in rvcalib_priors.keys() :
            if key in labels :
                theta_priors[key] = rvcalib_priors[key]

    if len(flare_priors) :
        for key in flare_priors.keys() :
            if key in labels :
                theta_priors[key] = flare_priors[key]

    return theta, labels, theta_priors



def get_theta_from_rv_priors(planet_priors, rvcalib_priors) :
    
    theta_planet, planet_labels = read_exoplanet_rv_params(planet_priors, output_theta_params=True)

    theta_calib, calib_labels = read_rvcalib_params(rvcalib_priors, output_theta_params=True)

    theta = np.concatenate((theta_planet, theta_calib), axis=0)
    labels = planet_labels + calib_labels

    theta_priors = {}
    for key in planet_priors.keys() :
        if key in labels :
            theta_priors[key] = planet_priors[key]
    for key in rvcalib_priors.keys() :
        if key in labels :
            theta_priors[key] = rvcalib_priors[key]

    return theta, labels, theta_priors


def read_exoplanet_rv_params(prior_dict, output_theta_params = False):

    param_ids = ['teff', 'ms', 'rs', 'rhos', 'n_planets']

    n_planets = prior_dict["n_planets"]['object'].value

    planet_param_ids = ['k', 'tc', 'per', 'ecc', 'w', 'rvsys', 'trend', 'quadtrend']

    for planet_index in range(int(n_planets)) :
        for i in range(len(planet_param_ids)):
            tmp_id = "{0}_{1:03d}".format(planet_param_ids[i], planet_index)
            param_ids.append(tmp_id)
    
    planet_params = {}
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            planet_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in planet_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(planet_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(planet_params.keys())
        for key in param_keys:
            error_key = "{0}_err".format(key)
            planet_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            planet_params[pdf_key] = prior_dict[pdf_key]
        return planet_params


def read_exoplanet_transit_rv_params(prior_dict, output_theta_params = False) :

    param_ids = ['teff', 'ms', 'rs', 'rhos', 'n_planets']

    n_planets = prior_dict["n_planets"]['object'].value

    planet_param_ids = ['transit', 'k', 'tc', 'per', 'ecc', 'w', 'rvsys', 'trend', 'quadtrend']
    planet_transit_param_ids = ['a','rp','b','inc','u0','u1']

    for planet_index in range(int(n_planets)) :
    
        planet_transit_id = "{0}_{1:03d}".format('transit', planet_index)
        
        is_planet_transit = prior_dict[planet_transit_id]['object'].value
    
        if is_planet_transit :
            for i in range(len(planet_transit_param_ids)):
                tmp_id = "{0}_{1:03d}".format(planet_transit_param_ids[i], planet_index)
                param_ids.append(tmp_id)

        for i in range(len(planet_param_ids)):
            tmp_id = "{0}_{1:03d}".format(planet_param_ids[i], planet_index)
            param_ids.append(tmp_id)
    
    planet_params = {}
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            planet_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in planet_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(planet_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(planet_params.keys())
        for key in param_keys:
            error_key = "{0}_err".format(key)
            planet_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            planet_params[pdf_key] = prior_dict[pdf_key]
        return planet_params



def read_exoplanet_params(prior_dict, planet_index=0, output_theta_params = False):
    
    param_ids = ['teff', 'ms', 'rs', 'rhos']
    
    #param_ids = ['per','tau','k','omega','ecc','rv0','lambda', 'vsini','a_R','inc','r_R','omega_rm','ldc']
    #prim_param_ids = ['ms', 'rs', 'tc', 'per', 'rp', 'b', 'u0', 'u1']
    #prim_param_ids = ['teff', 'ms', 'rs', 'k', 'tc', 'tp', 'per', 'a', 'rp', 'b', 'inc', 'ecc', 'w', 'u0', 'u1', 'rvsys', 'trend', 'quadtrend']
    #prim_param_ids = ['teff', 'ms', 'rs', 'k', 'tc', 'per', 'a', 'rp', 'inc', 'ecc', 'w', 'u0', 'u1', 'rvsys', 'trend', 'quadtrend']
    #prim_param_ids = ['k', 'tc', 'per', 'a', 'rp', 'inc', 'ecc', 'w', 'u0', 'u1', 'rvsys', 'trend', 'quadtrend']
    prim_param_ids = ['k', 'tc', 'a', 'per', 'b', 'inc', 'rp', 'ecc', 'w', 'u0', 'u1', 'rvsys', 'trend', 'quadtrend', 'transit']

    for i in range(len(prim_param_ids)):
        tmp_id = "{0}_{1:03d}".format(prim_param_ids[i], planet_index)
        param_ids.append(tmp_id)
    
    planet_params = {}
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            planet_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in planet_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(planet_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(planet_params.keys())
        for key in param_keys:
            error_key = "{0}_err".format(key)
            planet_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            planet_params[pdf_key] = prior_dict[pdf_key]
        return planet_params


def read_instrument_params(prior_dict, output_theta_params = False, inst_param_labels = ['u0', 'u1', 'rp']):
        
    instrument_params = {}
    param_ids = []
    
    n_instruments = prior_dict["n_instruments"]['object'].value
    
    for instrument_index in range(n_instruments) :
        for i in range(len(inst_param_labels)):
            tmp_id = "{}_inst_{:05d}".format(inst_param_labels[i], instrument_index)
            param_ids.append(tmp_id)
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            instrument_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in instrument_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(instrument_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(instrument_params.keys())
        """
        for key in param_keys:
            error_key = "{0}_err".format(key)
            instrument_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            instrument_params[pdf_key] = prior_dict[pdf_key]
        """
        return instrument_params


#intialize instrument parameters
def init_instrument_priors(n_instruments=1, inst_param_labels = ['u0', 'u1', 'rp'], ini_values = [0.2, 0.2, 0.001]) :
    """
        n_instruments : number of instruments
        inst_param_labels : list of instrument-related parameter labels
        ini_values: list of instrument-related parameter initial values
        """
    priors = {}

    n_instruments_dict = {}
    n_instruments_dict['type'] = "FIXED"
    n_instruments_dict['object'] = constant_parameter(n_instruments)
    priors['n_instruments'] = n_instruments_dict
    
    for j in range(n_instruments):
        for i in range(len(inst_param_labels)):
            inst_param_id = "{}_inst_{:05d}".format(inst_param_labels[i], j)
            out_dict = {}
            if inst_param_labels[i] == "u0" or inst_param_labels[i] == "u1" :
                l_value, u_value = 0, 1
            elif inst_param_labels[i] == "rp" :
                l_value, u_value = 0.00001, 1.0
                
            out_dict['type'] = "Uniform"
            out_dict['object'] = uniform_parameter(np.array([l_value, u_value]))
            out_dict['object'].set_value(float(ini_values[i]))
            
            priors[inst_param_id] = out_dict

    return priors


def read_starrot_gp_params(prior_dict, output_theta_params = False, spec_constrained=False) :
    
    if spec_constrained :
        param_ids = ['blong_mean', 'blong_white_noise', 'blong_amplitude', 'rv_mean', 'rv_white_noise', 'rv_amplitude', 'phot_mean', 'phot_white_noise', 'phot_amplitude', 'spec_decaytime', 'spec_smoothfactor', 'phot_decaytime', 'phot_smoothfactor', 'prot']
    else :
        param_ids = ['blong_mean', 'blong_white_noise', 'blong_amplitude', 'rv_mean', 'rv_white_noise', 'rv_amplitude', 'phot_mean', 'phot_white_noise', 'phot_amplitude', 'blong_decaytime', 'blong_smoothfactor', 'rv_decaytime', 'rv_smoothfactor', 'phot_decaytime', 'phot_smoothfactor', 'prot']

    gp_params = {}
    gp_params['param_ids'] = param_ids
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            gp_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in gp_params['param_ids'] :
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(gp_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(gp_params.keys())
        for key in param_ids :
            error_key = "{0}_err".format(key)
            gp_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            gp_params[pdf_key] = prior_dict[pdf_key]
        return gp_params



def get_gp_theta_from_priors(gp_priors) :

    gp_theta, gp_labels = read_starrot_gp_params(gp_priors, output_theta_params=True)
    
    gp_theta_priors = {}

    for key in gp_priors.keys() :
        if key in gp_labels :
            gp_theta_priors[key] = gp_priors[key]

    return gp_theta, gp_labels, gp_theta_priors


def read_phot_starrot_gp_params(prior_dict, output_theta_params = False):
    
    param_ids = ['mean', 'white_noise', 'amplitude', 'period', 'decaytime', 'smoothfactor']
    
    gp_params = {}
    
    for i in range(len(param_ids)):
        if param_ids[i] in prior_dict.keys() :
            param = prior_dict[param_ids[i]]
            gp_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in gp_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(gp_params[key])
                labels.append(key)
        return theta, labels
    else :
        param_keys = list(gp_params.keys())
        for key in param_keys:
            error_key = "{0}_err".format(key)
            gp_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(key)
            gp_params[pdf_key] = prior_dict[pdf_key]
        return gp_params


def update_prior_initial_guess_from_posterior(planet_priors_file, planet_posteriors_file, updated_priors_file=None) :

    if updated_priors_file is None :
        updated_priors_file = planet_priors_file

    planet_priors = read_priors(planet_priors_file)
    planet_posteriors = read_priors(planet_posteriors_file)

    outfile = open(updated_priors_file,"w")
    outfile.write("# Parameter_ID\tPrior_Type\tValues\n")

    for key in planet_priors.keys() :
    
        if ("_err" not in key) and ("_pdf" not in key) :
        
            new_value = planet_priors[key]['object'].value
            
            if planet_priors[key]['type'] == 'FIXED' :
                outfile.write("{}\tFIXED\t{:.10f}\n".format(key,planet_priors[key]['object'].value))
            elif planet_priors[key]['type'] == 'Normal' or  planet_priors[key]['type'] == 'Normal_positive':
                value_l = planet_priors[key]['object'].value_l
                value_u = planet_priors[key]['object'].value_u
                if key in planet_posteriors.keys() :
                    new_value = planet_posteriors[key]['object'].value
                    value_l = planet_posteriors[key]['object'].value_l
                    value_u = planet_posteriors[key]['object'].value_u
                outfile.write("{}\t{}\t{:.10f},{:.10f}\n".format(key,planet_priors[key]['type'],new_value,(value_l+value_u)/2))
            elif  planet_priors[key]['type'] == 'Uniform' :
                if key in planet_posteriors.keys() :
                    new_value = planet_posteriors[key]['object'].value
                value_l = planet_priors[key]['object'].value_l
                value_u = planet_priors[key]['object'].value_u
                outfile.write("{}\t{}\t{:.10f},{:.10f},{:.10f}\n".format(key,planet_priors[key]['type'],value_l,value_u,new_value))

    outfile.close()

    return updated_priors_file


def update_posteriors_file(posteriors_file, keys_to_update=[], types=[], values=[], values_l=[], values_u=[], updated_posteriors_file=None) :

    if updated_posteriors_file is None :
        updated_posteriors_file = posteriors_file

    posteriors = read_priors(posteriors_file)

    outfile = open(updated_posteriors_file,"w")
    outfile.write("# Parameter_ID\tPrior_Type\tValues\n")

    for key in posteriors.keys() :
    
        if ("_err" not in key) and ("_pdf" not in key) :
        
            if key in keys_to_update :
            
                idx = keys_to_update.index(key)
                
                if types[idx] is not None :
                    posteriors[key]['type'] = types[idx]
                if values[idx] is not None :
                    posteriors[key]['object'].value = values[idx]
                if values_l[idx] is not None :
                    posteriors[key]['object'].value_l = values_l[idx]
                if values_u[idx] is not None :
                    posteriors[key]['object'].value_u = values_u[idx]

            value = posteriors[key]['object'].value
            pdf_type = posteriors[key]['type']
            
            if pdf_type == 'FIXED' :
                outfile.write("{}\t{}\t{:.10f}\n".format(key,pdf_type,value))
            elif pdf_type == 'Normal' or  pdf_type == 'Normal_positive':
                value_l = posteriors[key]['object'].value_l
                value_u = posteriors[key]['object'].value_u
                outfile.write("{}\t{}\t{:.10f},{:.10f}\n".format(key,pdf_type,value,(value_l+value_u)/2))
            elif pdf_type == 'Uniform' :
                value_l = posteriors[key]['object'].value_l
                value_u = posteriors[key]['object'].value_u
                outfile.write("{}\t{}\t{:.10f},{:.10f},{:.10f}\n".format(key,pdf_type,value_l,value_u,value))

    outfile.close()

    return updated_posteriors_file
