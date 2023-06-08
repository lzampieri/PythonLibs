import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname
import pickle

def get_available_ref():
    labels = []
    with open( dirname( __file__ ) + '/gases_lib.pkl', 'rb') as handle:
        gases = pickle.load( handle )
    for g in gases:
        labels.append( g['label'].replace( '$', '' ).replace( '_', '' ) )
    return labels


def plot_with_ref( wl, A, xlim = None, ylim = None, withref = [] ):
    
    if( xlim is None ):
        xlim = [ np.min( wl ), np.max( wl ) ]

    if( ylim is None ):
        ylim = [ np.min( A ) - 0.2 * np.ptp( A ), np.max( A ) + 0.2 * np.ptp( A ) ]

    
    wl = np.array( wl )
    A = np.array( A )
    idx = ( wl > np.min( xlim ) ) * ( wl < np.max( xlim ) )
    
    # A[ A > np.max( ylim ) ] = 0
    # A[ A < np.min( ylim ) ] = 0

    plt.plot( wl[ idx ], A[ idx ] )

    plt.plot( xlim, [0,0], '--k', lw=1 )

    plt.xlim( np.flip( np.sort( xlim ) ) )

    plt.ylim( ylim )

    if( len( withref ) > 0 ):
        
        yticks, _ = plt.yticks()
        yticks  = yticks [ yticks <= np.max( ylim ) ]
        yticks  = yticks [ yticks >= np.min( ylim ) ]

        
        yampl = np.ptp( ylim ) / 5
        ymin = np.min( ylim ) - yampl
        plt.ylim( ymin, np.max( ylim ) )
        plt.plot( xlim, [ ymin + yampl, ymin + yampl ], '-k', lw=1, label='Data' )

        with open( dirname( __file__ ) + '/gases_lib.pkl', 'rb') as handle:
            gases = pickle.load( handle )

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
        colors = np.repeat( colors, int( len( gases ) / len( colors ) ) + 2 )
        for g, color in zip( gases, colors ):
            label = g['label'].replace( '$', '' ).replace( '_', '' )
            if( label in withref ):
                expected_spectra = np.interp( wl, g['spectrum'][0,:], g['spectrum'][1,:], left = 0, right = 0 )
                rescaled_spectra = expected_spectra * yampl * 0.9 / np.max( expected_spectra )
                main_idx = np.where( (expected_spectra > 0.4 * np.max( expected_spectra )) * idx )[0]
                main_idx_seq = np.split( main_idx, np.where( np.diff( main_idx ) > 1 )[0] + 1 )

                plt.plot( wl[ idx ], rescaled_spectra[ idx ] + ymin, alpha = 0.3, color=color )

                for seq in main_idx_seq:
                    plt.plot( wl[ seq ], rescaled_spectra[ seq ] + ymin, color=color )
                plt.plot( [], [], color=color, label = g['label'] )

        plt.yticks( yticks )

        plt.legend()

def get_ref_spectrum( gas ):
    with open( dirname( __file__ ) + '/gases_lib.pkl', 'rb') as handle:
        gases = pickle.load( handle )

    for g in gases:
        label = g['label'].replace( '$', '' ).replace( '_', '' )
        if ( label == gas ):
            return g['spectrum'][0,:], g['spectrum'][1,:]

