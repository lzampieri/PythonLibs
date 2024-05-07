import numpy as np
import pandas as pd
from uncertainties import ufloat
from . import massiveOES
from collections import OrderedDict

def fitAndGetPars( wl, int, autocrop = True ):
    """
    Fit data with massiveOES and return fit results
    ---
    Parameters:
    wl: wavelength, numpy array containing the wavelengths
    int: intensity, numpy array containing the intensities
    autocrop: automatically crop data to the interval 360-383. Default: True
    ---
    Return:
    fit parameters, in a dictionary
    """
    interface = prepareInterface( wl, int, autocrop )
    fit( interface )
    return fit_pars( interface )

def prepareInterface( wl, int, autocrop = True ):
    """
    Prepare the interface to fit via MassiveOES
    ---
    Parameters:
    wl: wavelength, numpy array containing the wavelengths
    int: intensity, numpy array containing the intensities
    autocrop: automatically crop data to the interval 360-383. Default: True
    ---
    Return:
    massiveOES interface
    """

    # Eventually crop spectrum
    if( autocrop ):
        idx = ( wl > 360 ) & ( wl < 383 )
        wl = np.array(wl)[idx]
        int = np.array(int)[idx]
        print(f"Data autocropped to the interval 360-383 (from {len(idx)} to {len(int)} datapoints)")

    # Init massiveOES interface
    interface = massiveOES.MeasuredSpectra(
        spectra = OrderedDict( { 0: { 'spectrum': int }})
        )

    # Add species    
    interface.add_specie( massiveOES.SpecDB('N2CB.db'), 0 )
    # Other transitions available are 'C2_swan.db', 'N2CB.db', 'N2PlusBX.db', 'NHAX.db', 'NOBX.db', 'OHAX.db',

    # Setup fit params
    interface.spectra[0]['params']['slitf_gauss'].set( value = 0.1, min = 0, max = 0.4, vary = True )
    interface.spectra[0]['params']['slitf_lorentz'].set( value = 0.1, min = 0, max = 0.4, vary = True )

    min_wl = np.min( wl )
    interface.spectra[0]['params']['wav_start'].set( min = min_wl - 3, value = min_wl, max = min_wl + 3, vary = True )
    interface.spectra[0]['params']['wav_step'].set( np.ptp( wl ) / np.size( wl ), vary = False )
    interface.spectra[0]['params']['wav_2nd'].set( 0.0, vary = False )

    interface.spectra[0]['params']['baseline'].set( 0.0, min = -2, max = 2, vary = True )
    interface.spectra[0]['params']['baseline_slope'].set( 0.0, vary = False )

    return interface

def params_list( interface ):
    """
    List of fit parameters on a interface
    ---
    Parameters:
    interface: massiveOES interface
    ---
    Return:
    list of fit parameters keys
    """
    return interface.spectra[0]['params'].keys()

def setParam( interface, param, value = None, vary = None, min = None, max = None ):
    """
    Set new values for a parameter
    Arguments with None value are ignored
    ---
    Parameters:
    interface: massiveOES interface
    param: parameter key
    value: float, parameters value, default None
    vary: boolean, if the parameter should be varied during the fit or be fixed, default None
    min: float, min value for the parameter, default None
    max: float, max value for the parameter, default None
    """
    if( value == None ):
        value = interface.spectra[0]['params'][param].value
    if( vary  == None ):
        vary  = interface.spectra[0]['params'][param].vary
    if( min   == None ):
        min   = interface.spectra[0]['params'][param].min
    if( max   == None ):
        max   = interface.spectra[0]['params'][param].max
    
    interface.spectra[0]['params'][param].set( value = value, vary = vary, min = min, max = max )

def fit( interface, show_results = True ):
    """
    Perform the fit
    ---
    Parameters:
    interface: massiveOES interface
    show_results: boolean, if the fit results should be displayed, default True
    ---
    Return:
    fit results, in a dictionary struct
    """
    interface.fit( 0 )

    results = fit_pars( interface )

    if( show_results ):
        display( pd.DataFrame.from_records([ results ] ) )
    
    return results

def fit_pars( interface ):
    """
    Extract the fit parameters
    ---
    Parameters:
    interface: massiveOES interface
    ---
    Return:
    fit results, in a dictionary
    """
    
    def par( k ):
        if( interface.spectra[0]['params'][k].vary ):
            if( interface.spectra[0]['params'][k].stderr == None ):
                return ufloat( interface.spectra[0]['params'][k].value, 0 )
            return ufloat( interface.spectra[0]['params'][k].value, interface.spectra[0]['params'][k].stderr )
        else:
            return interface.spectra[0]['params'][k].value
        
    output = { k : par( k ) for k in interface.spectra[0]['params'].keys() }
    output['fitted_spectrum'] = get_fitted_spectrum( interface )
    return output

def get_fitted_spectrum( interface ):
    """
    Extract the fitted spectrum
    ---
    Parameters:
    interface: massiveOES interface
    ---
    Return:
    fitted spectrum, in a dictionary: {
        'wl': wavelength,
        'int': fitted intensity
    }
    """

    sim_spec = massiveOES.puke_spectrum( interface.spectra[0]['params'], sims = interface.simulations )

    wl = np.array( sim_spec.x )
    int = np.array( sim_spec.y )
    wl_original = interface.get_measured_spectrum(0).x
    lims = [ 2 * wl_original[0] - wl_original[1], 2 * wl_original[-1] - wl_original[-2] ]

    idx = ( wl > np.min( lims ) ) * ( wl < np.max( lims ) )

    return { 
        'wl': wl[idx],
        'int': int[idx]
    }
