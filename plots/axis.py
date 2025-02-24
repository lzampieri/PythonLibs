import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def time():
    xlims = plt.xlim()

    max_t = np.max( np.abs( xlims ) )

    degree = - np.log10( max_t )
    degree_abs = 0 if degree < 0 else degree
    degree_mul = ceil( degree_abs / 3 ) * 3

    ticks, _ = plt.xticks()
    if( len( ticks ) > np.ptp( ticks * ( 10 ** degree_mul ) ) + 1 ):
        plt.xticks( ticks, [f"{t:.1f}" for t in ticks * ( 10 ** degree_mul )] )
    else:
        plt.xticks( ticks, [f"{t:.0f}" for t in ticks * ( 10 ** degree_mul )] )

    prefix = {
        0: '',
        3: 'm',
        6: 'μ',
        9: 'n',
        12: 'p'
    }

    plt.xlabel( f"Time [{prefix[degree_mul]}s]" )

    
def voltage():
    if( np.ptp( plt.ylim() ) > 2000 ):
        ticks, _ = plt.yticks()
        if( len( ticks ) > np.ptp( ticks / 1000 ) + 1 ):
            plt.yticks( ticks, [f"{t:.1f}" for t in ticks / 1000 ] )
        else:
            plt.yticks( ticks, [f"{t:.0f}" for t in ticks / 1000 ] )
        plt.ylabel( f"Voltage [kV]" )

    else:
        plt.ylabel( f"Voltage [V]" )

def Voft():
    time()
    voltage()

def spectrum():
    plt.xlabel( f"Wavelength [nm]" )
    plt.ylabel( f"Counts [a.u.]" )
    ticks, _ = plt.yticks()
    plt.yticks( ticks, [ '' for t in ticks ] )

def FTIR():
    plt.xlabel( r"Wavelength [cm${}^{-1}$]" )
    plt.ylabel( r"Absorbance" )
    
    # Revert axis if necessary
    lims = plt.xlim()
    if( lims[0] < lims[1] ):
        plt.gca().invert_xaxis()

def FTIR_areas():
    plt.ylabel( r"Integrated absorbance [$cm^{-1}$]" )

def FTIR_ppm():
    plt.ylabel( r"Concentration [ppm]" )
    
def GC():
    plt.xlabel( r"Retention time [min]" )
    plt.ylabel( r"Intensity [counts]" )

def addToLegend( text ):
    plt.plot( [], [], 'k', alpha = 0, label = text )

def symmetrical( x = True, y = True ):
    if( x ):
        plt.xlim( np.array([-1,1]) * np.max( np.abs( plt.xlim() ) ) )
    if( y ):
        plt.ylim( np.array([-1,1]) * np.max( np.abs( plt.ylim() ) ) )

def cartesian_axis( show_h = True, show_v = True ):
    xlim = plt.xlim()
    ylim = plt.ylim()

    if( xlim[0] * xlim[1] < 0 and show_v):
        ylim_axis = ylim + np.array([1,-1]) * 0.03 * np.ptp( ylim )
        plt.plot( [0,0], ylim_axis, '--k' )

    if( ylim[0] * ylim[1] < 0 and show_h):
        xlim_axis = xlim + np.array([1,-1]) * 0.03 * np.ptp( xlim )
        plt.plot( xlim_axis, [0,0], '--k' )

    plt.xlim(xlim)
    plt.ylim(ylim)

def tilt_xlabels( angle = 30 ):
    plt.xticks( rotation = angle, ha = 'right' if angle <= 45 else 'center' )

def draft():
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.annotate( "DRAFT", [np.mean( xlim ), np.min( ylim ) + 0.2 * np.ptp( ylim )], ha='center', fontsize=80, color = 'red', weight = 'bold' )
    plt.xlim(xlim)
    plt.ylim(ylim)
