import numpy as np
import uncertainties
from uncertainties import unumpy
from standard_imports import mean, n
from scipy import ndimage
from math import floor
from uncertainties.core import Variable

def zero_crossings( ampl, time = [], with_uncertainties = True, low_threshold = 0.3, high_threshold = 0.5, expected_sharp = False, return_slopes = False ):

    if( len( time ) == 0 ):
        time = np.arange( len( ampl ) )

    assert len( time ) == len( ampl )

    thresh = { 1: np.max( ampl ) * low_threshold, -1: -np.min( ampl ) * low_threshold }
    out_thresh = { 1: np.max( ampl ) * high_threshold, -1: -np.min( ampl ) * high_threshold }

    hits_t = []
    slopes = []

    temp_i = []
    xs = np.array( time )
    ys = np.array( ampl )

    sign = 0
    for i in range( len( ampl ) ):
        if( ( sign == 0 ) and ( abs( ampl[i] ) > out_thresh[1] ) ):
            sign = np.sign( ampl[i] )
        
        if( sign != 0 ):
            if( sign * ampl[i] < thresh[sign] ):
                # If below the threshold, i.e. near zero, consider points for linear fit
                temp_i.append( i )
            if( - sign * ampl[i] > out_thresh[-sign] ):
                # If outside the threshold:
                fittemp_i = temp_i

                # If the transition is expected to be sharp, include also an extra point after and before
                if( expected_sharp ):
                    if( np.min( temp_i ) - 1 > 0 ):
                        fittemp_i = [ np.min( temp_i ) - 1 ] + fittemp_i
                    if( np.max( temp_i ) + 1 < len( xs ) - 1 ):
                        fittemp_i = fittemp_i + [ np.max( temp_i ) + 1 ]
                
                # If enough data
                if( len( fittemp_i ) > 2 or ( len( fittemp_i ) == 2 and expected_sharp ) ):
                    try:
                        p, cov = np.polyfit( xs[fittemp_i], ys[fittemp_i], 1, cov=True)
                        # Fit to find zero crossings
                        ( m, q ) = uncertainties.correlated_values( p, cov )
                        hits_t.append( - q / m )
                        slopes.append( m )
                    except:
                        print( f"Skipped a zero estimation due to fit errors!")
                temp_i = []
                sign = sign * -1
            # if( ampl[i] > 0 ):
            #     print( f"I'm positive at {time[i]} with {len( temp_i ) } points considered" )
            #     print( - sign * ampl[i], out_thresh[sign], - sign * ampl[i] > out_thresh[sign] )
    hits_t = np.array( hits_t )
    slopes = np.array( slopes )

    if( not with_uncertainties ):
        hits_t = unumpy.nominal_values( hits_t )

    if( return_slopes ):
        return hits_t, slopes
    
    return hits_t

def frequency( ampl, time = [], with_uncertainties = True ):
    return 1 / period( ampl, time, with_uncertainties )

def period( ampl, time = [], with_uncertainties = True ):
    hits_t = zero_crossings( ampl, time, with_uncertainties )

    if( len( hits_t ) % 2 == 0 ):
        hits_t = hits_t[:-1]

    differences_t = hits_t[1:] - hits_t[:-1]

    return np.mean( differences_t ) * 2

def phase_t( ampl, time = [] ):
    # Find a odd number of zero crossings
    hits_t = zero_crossings( ampl, time )
    if( len( hits_t ) % 2 == 0 ):
        hits_t = hits_t[:-1]

    # Estimate period:
    T = mean( hits_t[1:] - hits_t[:-1] )
    phase_t = mean( hits_t - T * np.arange( len( hits_t ) ) )

    while( phase_t.n < 0 ):
        phase_t += T.n
    while( phase_t.n > T.n ):
        phase_t -= T.n

    return phase_t

def time_step( time ):
    return ( time[ 200 ] - time[ 100 ] ) / 100

def shift_waveform( waveform, shift ):
    shifted_0 = ndimage.shift( waveform, -shift.n, mode = 'nearest' )
    shifted_p = ndimage.shift( waveform, -shift.n + shift.s, mode = 'nearest' )
    shifted_m = ndimage.shift( waveform, -shift.n - shift.s, mode = 'nearest' )
    output = np.zeros_like( waveform, dtype=Variable )
    for i in range( len( waveform ) ):
        output[i] = mean( [ shifted_0[i], shifted_p[i], shifted_m[i] ] )
    return output

def periods_crop( ampl, time, T = -1, getN = False ):
    if( T < 0 ):
        T = period( n( ampl ), time )
    N_periods = floor( ( time[-3] - time[2] ) / n( T ) )
    assert N_periods > 0, "Less than a period in the provided signal"
    start_idx = np.max( np.where( time < np.mean( time ) - N_periods * T / 2 )[0] )
    end_idx = np.max( np.where( time < np.mean( time ) + N_periods * T / 2 )[0] )

    if( getN ):
        return ampl[ start_idx : end_idx + 1 ], time[ start_idx : end_idx + 1 ], N_periods

    return ampl[ start_idx : end_idx + 1 ], time[ start_idx : end_idx + 1 ]