import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from plots import splt, adv_plt
from tqdm.auto import tqdm
from uncertainties import correlated_values
from calibrations import calibrations

propane_main_satval = 254
propane_first_maxval = 85
propane_second_minval = 10

def compute_propane_second( wn, sp, plot = False ):

    # Propane-related area
    propane_second_idx = ( wn > 1400 ) * ( wn < 1550 )

    # propane_second-integrating area
    propane_second_int_idx = ( wn > 1425 ) * ( wn < 1500 )

    integral = np.trapz( sp[ propane_second_int_idx ],  wn[ propane_second_int_idx ] )

    if( plot ):
        plt.plot( wn[ propane_second_idx ], sp[ propane_second_idx ] )
        plt.fill_between( wn[ propane_second_int_idx ], 0, sp[ propane_second_int_idx ], alpha = 0.3, label = f"propane_second: {integral:.4f}" )

    return integral

def compute_propane_main( wn, sp, plot = False ):

    # propane_main-related area
    propane_main_idx = ( wn > 2900 ) * ( wn < 3020 )

    # propane_main-integrating area
    propane_main_int_idx = ( wn > 2945 ) * ( wn < 2990 )

    integral = np.trapz( sp[ propane_main_int_idx ],  wn[ propane_main_int_idx ] )

    if( plot ):
        plt.plot( wn[ propane_main_idx ], sp[ propane_main_idx ] )
        plt.fill_between( wn[ propane_main_int_idx ], 0, sp[ propane_main_int_idx ], alpha = 0.3, label = f"propane_main: {integral:.4f}" )

    if( np.max( sp[ propane_main_int_idx ] ) > 5.5 ):
        if( plot ):
            idx_sat = np.where( sp[ propane_main_int_idx ] > 5.5 )[0]
            plt.plot( wn[ propane_main_int_idx ][ idx_sat ], sp[ propane_main_int_idx ][idx_sat], 'or' )
        return propane_main_satval
    
    return integral

def calibrate_propane( propane_second, propane_main ):

    propane_second_ppm = calibrations.calibrate( "propane_secondpeak_FTIR_10m", propane_second )
    propane_main_ppm = calibrations.calibrate( "propane_mainpeak_FTIR_10m", propane_main )

    propane_avg_ppm = np.mean( [propane_second_ppm, propane_main_ppm], axis = 0 )

    idx_only_main = np.where( np.array( propane_second ) < propane_second_minval )[0]
    propane_avg_ppm[ idx_only_main ] = propane_main_ppm[ idx_only_main ]

    idx_only_second = np.where( np.array( propane_main ) > propane_first_maxval )[0]
    propane_avg_ppm[ idx_only_second ] = propane_second_ppm[ idx_only_second ]

    return {
        'propane_second_ppm': propane_second_ppm,
        'propane_main_ppm': propane_main_ppm,
        'propane_ppm': propane_avg_ppm
    }


def estimate_areas( wn, spectra, quiet = False, plot = False, calibrate = True ):

    if( spectra.ndim == 1 ):
        spectra = np.array( [ spectra ] )

    areas = { 'propane_second': [], 'propane_main': [] }

    if( isinstance( plot, bool ) ):
        plot = list( areas.keys() ) if plot == True else []

    flip = False
    if( wn[-1] < wn[0] ):
        wn = np.flip( wn )
        flip = True

    if( plot ):
        splt.init_bytot( len( spectra ) )
    
    for i, sp in ( enumerate( spectra ) if quiet else tqdm( list( enumerate( spectra ) ), leave = False ) ):
        sp = spectra[i,:]

        if( plot ):
            splt.next()

        if( flip ):
            sp = np.flip( sp )

        propane_second = compute_propane_second( wn, sp, 'propane_second' in plot )
        propane_main = compute_propane_main( wn, sp, 'propane_main' in plot )

        areas['propane_second'].append( propane_second )
        areas['propane_main'].append( propane_main )

        if( plot ):
            plt.legend()
            plt.title( i )

        # if( i > 10 ):
            # break

    if( calibrate ):
        areas.update( calibrate_propane( areas['propane_second'], areas['propane_main'] ) )

    for k in areas.keys():
        if( len( areas[k] ) == 1 ):
            areas[k] = areas[k][0]
    
    return areas