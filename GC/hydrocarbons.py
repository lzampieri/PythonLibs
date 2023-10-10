import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from plots import splt, adv_plt
from tqdm.auto import tqdm
from uncertainties import correlated_values
from calibrations import calibrations

def compute_propane( time, sp, plot = False ):

    # Propane-related area
    propane_idx = ( time > 0.25 ) * ( time < 0.5 )

    # Baseline-related area
    baseline_idx = propane_idx * ( ( time < 0.3 ) + ( time > 0.4 ) )

    # Propane-integrating area
    propane_int_idx = ( time > 0.3 ) * ( time < 0.36 )

    # Baseline fitting
    p, pcov = np.polyfit( time[ baseline_idx ], sp[ baseline_idx ], 3, cov = True )
    up = correlated_values( p, pcov )

    # Compute integral
    integral = np.trapz( sp[ propane_int_idx ] - np.polyval( up, time[ propane_int_idx ] ),  time[ propane_int_idx ] )

    # Plot
    if( plot ):
        plt.plot( time[ propane_idx ], sp[ propane_idx ] )
        plt.plot( time[ propane_idx ], adv_plt.unp_n( np.polyval( up, time[ propane_idx ] ) ) )
        plt.fill_between( time[ propane_int_idx ], adv_plt.unp_n( np.polyval( up, time[ propane_int_idx ] ) ), sp[ propane_int_idx ], alpha = 0.3, label = f"propane: {integral:.4f}" )

    return integral

def estimate_areas( time, spectra, quiet = False, plot = False, calibrate = True ):

    if( spectra.ndim == 1 ):
        spectra = np.array( [ spectra ] )

    areas = { 'propane': [] }
    if( isinstance( plot, bool ) ):
        plot = list( areas.keys() ) if plot == True else []

    flip = False
    if( time[-1] < time[0] ):
        time = np.flip( time )
        flip = True

    if( plot ):
        splt.init_bytot( spectra.ndim > 1 )
    
    for i, sp in ( enumerate( spectra ) if quiet else tqdm( list( enumerate( spectra ) ), leave=False ) ):
        sp = spectra[i,:]

        if( plot ):
            splt.next()

        if( flip ):
            sp = np.flip( sp )

        propane = compute_propane( time, sp, 'propane' in plot )

        if( len( spectra ) > 1 ):
            areas['propane'].append( propane )
        else:
            areas['propane'] = propane

        if( plot ):
            plt.legend()
            plt.title( i )

        # if( i > 10 ):
            # break

    if( calibrate ):
        areas['propane_ppm'] = calibrations.calibrate( "propane_GC", areas['propane'] )

    
    return areas