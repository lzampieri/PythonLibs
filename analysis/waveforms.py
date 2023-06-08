import numpy as np
import uncertainties
from uncertainties import unumpy

def zero_crossings( ampl, time = [], with_uncertainties = True, low_threshold = 0.3, high_threshold = 0.5 ):

    if( len( time ) == 0 ):
        time = np.arange( len( ampl ) )

    assert len( time ) == len( ampl )

    thresh = np.abs( np.max( np.abs( ampl ) ) ) * low_threshold
    out_thresh = np.abs( np.max( np.abs( ampl ) ) ) * high_threshold


    hits_t = []
    temp = { 'x': [], 'y': [] }
    sign = 0
    for i in range( len( ampl ) ):
        if( ( sign == 0 ) and ( abs( ampl[i] ) > out_thresh ) ):
            sign = np.sign( ampl[i] )
        
        if( sign != 0 ):
            if( sign * ampl[i] < thresh ):
                # If below the threshold, i.e. near zero, consider points for linear fit
                temp['x'].append( time[i] )
                temp['y'].append( ampl[i] )
            if( - sign * ampl[i] > out_thresh ):
                # If outside the threshold:
                if( len( temp['x'] ) > 2 ):
                    # If enough data
                    try:
                        p, cov = np.polyfit( temp['x'], temp['y'], 1, cov=True)
                        # Fit to find zero crossings
                        ( m, q ) = uncertainties.correlated_values( p, cov )
                        hits_t.append( - q / m )
                    except:
                        print( f"Skipped a zero estimation due to fit errors!")
                temp = { 'x': [], 'y': [] }
                sign = sign * -1
    hits_t = np.array( hits_t )

    if( not with_uncertainties ):
        return unumpy.nominal_values( hits_t )
    return hits_t

def frequency( ampl, time = [], with_uncertainties = True ):
    hits_t = zero_crossings( ampl, time, with_uncertainties )

    if( len( hits_t ) % 2 == 0 ):
        hits_t = hits_t[:-1]

    differences_t = hits_t[1:] - hits_t[:-1]

    return 1 / np.mean( differences_t ) / 2
