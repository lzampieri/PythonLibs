import numpy as np
from . import zeros
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values as unp_n
from uncertainties import ufloat

def n( x ):
    return x.n if 'uncertainties' in str( type( x ) ) else x
def s( x ):
    return x.s if 'uncertainties' in str( type( x ) ) else 0

def compute_lissajous( time, voltage_minus_C_voltage, charge, crop = True, N_periods = 0, plot = False, return_cropped_idx = False, conservative_errors = True ):
    
    assert crop or N_periods, f"🚨🚨 Either crop must be true or the number of periods must be provided! 🚨🚨"

    voltage = voltage_minus_C_voltage

    if( crop ):
        zlist = zeros.zero_crossings( voltage, time )

        # If no enought zeros...
        assert len( zlist ) > 2, f"🚨🚨 Not enought zeros in the waveform! 🚨🚨"

        # Assert a odd number of zeros
        if( len( zlist ) % 2 == 0 ):
            zlist = zlist[:-1]
        N_periods = int( ( len( zlist ) - 1 ) / 2 )
        period =  np.mean( zlist[1:] - zlist[:-1] ) * 2

        # Compute time limits
        central_time = np.mean( time )
        time_limits = unp_n( central_time + np.array( [ -0.5, 0.5 ] ) * N_periods * period )
        idx = ( time >= time_limits[0] ) * ( time < time_limits[1] )

        # Crop data
        time = time[ idx ]
        voltage = voltage[ idx ]
        charge = charge[ idx ]
    else:
        period = ( time[-1] - time[0] ) / N_periods

    # Compute the integral
    integral = np.trapz( voltage, charge ) / N_periods
    print(f"Computing integral on {N_periods} periods")

    # If conservative errors are required,
    # integrals are computed in each period
    # and their std is added to integral error
    if( conservative_errors and N_periods > 1 ):
        integrals = []

        idx_to_split = list( range( len( time ) ) )

        intervals = np.linspace( np.min( idx_to_split ), np.max( idx_to_split ), N_periods + 1, endpoint = True, dtype = int )
        for i in range( N_periods ):
            integrals.append( np.trapz (
                voltage[ intervals[i] : intervals[i + 1] ],
                charge[ intervals[i] : intervals[i + 1] ]
            ) )


        integral = ufloat(
            n( integral ),
            np.sqrt( s( integral ) ** 2 + np.std( n( integral ) ) ** 2 )
        )

    # Compute power
    power = integral / period

    if( plot ):
        plot_lissajous( voltage, charge, power, N_periods=N_periods )

    if( return_cropped_idx ):
        return power, idx, intervals # TODO remove
    return power

def plot_lissajous( voltage, charge, label = None, power = 0, forceClosure = True, N_periods = 0 ):

    if( forceClosure ):
        voltage = np.append( voltage, [ voltage[0] ] )
        charge = np.append( charge, [ charge[0] ] )

    plt.plot( voltage, charge )
    plt.fill_between( voltage, charge, alpha = 0.3, label = label if label else ( fr"${power:.1fL}$ W" if power else None ) )
    plt.xlabel( 'Voltage [V]' )
    plt.ylabel( 'Charge [C]' )

    if( N_periods ):
        plt.plot( [], [], 'w', label=fr"Averaged over {N_periods} periods" )
    if( power ):
        plt.legend()
