import numpy as np
import pandas as pd
from uncertainties import ufloat
from . import massiveOES
from collections import OrderedDict

def fitAndGetPars( wl, int ):
    interface = prepareInterface( wl, int )
    fit( interface )
    return fit_pars( interface )

def prepareInterface( wl, int ):

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
    return interface.spectra[0]['params'].keys()

def setParam( interface, param, value = None, vary = None, min = None, max = None ):
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
    interface.fit( 0 )

    results = fit_pars( interface )

    if( show_results ):
        display( pd.DataFrame.from_records([ results ] ) )
    
    return results

def fit_pars( interface ):
    
    def par( k ):
        if( interface.spectra[0]['params'][k].vary ):
            return ufloat( interface.spectra[0]['params'][k].value, interface.spectra[0]['params'][k].stderr )
        else:
            return interface.spectra[0]['params'][k].value
        
    return { k : par( k ) for k in interface.spectra[0]['params'].keys() }

def get_fitted_spectrum( interface ):

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
