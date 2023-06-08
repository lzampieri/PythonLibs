import numpy as np
from uncertainties import *
from uncertainties.unumpy import nominal_values as unp_n

def between( y, x, xlim ):
    idx = ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

    return np.array( y )[idx]

def compute_average( y, x, xlim ):
    data = unp_n( between( y, x, xlim ) )
    return ufloat( np.mean( data ), np.std( data ) )

def compute_averages( ys_dict, x, xlim ):
    output = {}

    for k, v in ys_dict.items():
        output[ k ] = compute_average( v, x, xlim )

    return output