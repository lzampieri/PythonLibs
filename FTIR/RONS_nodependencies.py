import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from uncertainties import correlated_values

def compute_ozone( wn, sp, plot = False, plot_with_real_wn = False ):
    # Ozone-related area
    # ozone_idx = ( wn > 1000 ) * ( wn < 1150 )
    ozone_idx = ( wn > 950 ) * ( wn < 1100 )
    ozone_wn = wn[ ozone_idx ]
    ozone_0wn = ozone_wn - ozone_wn[ 0 ]
    ozone_sp = sp[ ozone_idx ]

    # Detect if Ozone is up or down
    ozone_central_idx = np.max( np.where( wn < 1057 )[0] )
    ozone_a, _, _ = np.polyfit( wn[ ozone_central_idx - 4 : ozone_central_idx + 4 ], sp[ ozone_central_idx - 4 : ozone_central_idx + 4 ], 2 )
    ozone_direction = -1 if ozone_a > 0 else +1

    # Fitting function
    ozone_fitfun = lambda x, m, q, A1, s1, A2, s2 : A1 * np.exp( - ( x - 1057 + ozone_wn[ 0 ] )**2 / s1 ) + A2 * np.exp( - ( x - 1030 + ozone_wn[ 0 ] )**2 / s2 ) + m * x + q
    p0 = [ ( ozone_sp[-1] - ozone_sp[0] ) / ( ozone_wn[-1] - ozone_wn[0] ), ozone_sp[0], np.ptp( ozone_sp ) * ozone_direction, 15, np.ptp( ozone_sp ) * ozone_direction, 100 ]
    bounds = (
        # m,       q,       A1,      s1,      A2,      s2,    
        [ -np.inf, -np.inf, -np.inf,  0     , -np.inf,  0     ],
        [  np.inf,  np.inf,  np.inf,  1000  ,  np.inf,  1000  ]
    )
    
    try:
        p, pcov = curve_fit( ozone_fitfun, ozone_0wn, ozone_sp, p0, bounds=bounds )
    except (RuntimeError, RuntimeWarning):
        p = p0
        p[2] = 0
        p[4] = 0
        pcov = np.zeros( ( len( p0 ), len( p0 ) ) )

    ( m, q ) = correlated_values( p[0:2], pcov[0:2,0:2] )

    # Ozone-integrating area
    ozone_int_idx = ( wn > 1000 ) * ( wn < 1070 )
    # ozone_int_idx = ( wn > 990 ) * ( wn < 1070 )
    ozone_int_wn = wn[ ozone_int_idx ] - ozone_wn[ 0 ]
    integral = np.trapz( sp[ ozone_int_idx ] - m * ozone_int_wn - q,  wn[ ozone_int_idx ] )

    fit_info = { 'm': p[0], 'x': np.mean( ozone_int_wn ) + ozone_wn[ 0 ], 'y': m * np.mean( ozone_int_wn ) + q }

    if( plot ):
        plt.plot( ozone_0wn, ozone_sp )
        # plt.plot( fit_info['x'] - ozone_wn[ 0 ], fit_info['y'], 'dk' )
        plt.fill_between( ozone_int_wn, p[0] * ozone_int_wn + p[1], sp[ ozone_int_idx ], alpha = 0.3, label = f"Ozone: {integral:.4f}" )
    if( plot_with_real_wn ):
        plt.plot( ozone_0wn + ozone_wn[ 0 ], ozone_sp )
        plt.fill_between( ozone_int_wn + ozone_wn[ 0 ], p[0] * ozone_int_wn + p[1], sp[ ozone_int_idx ], alpha = 0.3, label = f"Ozone: {integral:.4f}" )

    return integral, fit_info

def compute_N2O( wn, sp, plot = False, plot_with_real_wn = False ):
    # N2O-related area
    N2O_idx = ( wn > 2150 ) * ( wn < 2285 )
    N2O_wn = wn[ N2O_idx ]
    N2O_0wn = N2O_wn - N2O_wn[ 0 ]
    N2O_sp = sp[ N2O_idx ]

    # Detect if N2O is up or down
    N2O_central_idx = np.max( np.where( wn < 2236 )[0] )
    N2O_a, _, _ = np.polyfit( wn[ N2O_central_idx - 4 : N2O_central_idx + 4 ], sp[ N2O_central_idx - 4 : N2O_central_idx + 4 ], 2 )
    N2O_direction = -1 if N2O_a > 0 else +1

    # Fitting function
    N2O_fitfun = lambda x, m, q, A1, s1, A2, s2 : A1 * np.exp( - ( x - 2238 + N2O_wn[ 0 ] )**2 / s1 ) + A2 * np.exp( - ( x - 2210 + N2O_wn[ 0 ] )**2 / s2 ) + m * x + q
    p0 = [ ( N2O_sp[-1] - N2O_sp[0] ) / ( N2O_wn[-1] - N2O_wn[0] ), N2O_sp[0], np.ptp( N2O_sp ) * N2O_direction, 10, np.ptp( N2O_sp ) * N2O_direction, 50 ]
    bounds = (
        # m,       q,       A1,      s1,      A2,      s2,    
        [ -np.inf, -np.inf, -np.inf,  0     , -np.inf,  0     ],
        [  np.inf,  np.inf,  np.inf,  500  ,  np.inf,  2000  ]
    )
    
    try:
        p, pcov = curve_fit( N2O_fitfun, N2O_0wn, N2O_sp, p0, bounds=bounds )
    except (RuntimeError, RuntimeWarning):
        p = p0
        p[2] = 0
        p[4] = 0
        pcov = np.zeros( ( len( p0 ), len( p0 ) ) )

    ( m, q ) = correlated_values( p[0:2], pcov[0:2,0:2] )

    # N2O-integrating area
    N2O_int_idx = ( wn > 2190 ) * ( wn < 2250 )
    N2O_int_wn = wn[ N2O_int_idx ] - N2O_wn[ 0 ]
    integral = np.trapz( sp[ N2O_int_idx ] - m * N2O_int_wn - q,  wn[ N2O_int_idx ] )

    fit_info = { 'm': p[0], 'x': np.mean( N2O_int_wn ) + N2O_wn[ 0 ], 'y': m * np.mean( N2O_int_wn ) + q }

    if( plot ):
        plt.plot( N2O_0wn, N2O_sp )
        plt.fill_between( N2O_int_wn, p[0] * N2O_int_wn + p[1], sp[ N2O_int_idx ], alpha = 0.3, label = f"N2O: {integral:.4f}" )
    if( plot_with_real_wn ):
        plt.plot( N2O_0wn + N2O_wn[ 0 ], N2O_sp )
        plt.fill_between( N2O_int_wn + N2O_wn[ 0 ], p[0] * N2O_int_wn + p[1], sp[ N2O_int_idx ], alpha = 0.3, label = f"N2O: {integral:.4f}" )

    return integral, fit_info

# def compute_NO2_midpoint_interpolation( wn, sp, O3_baseline, N2O_baseline, plot = False ):
#     # Select wide area to draw
#     draw_idx = ( wn > 1400 ) * ( wn < 1800 )
#     draw_wn = wn[ draw_idx ]
#     draw_sp = sp[ draw_idx ]


#     # Select NO2 second peak area
#     NO2_idx = ( wn > 1585 ) * ( wn < 1610 )
#     NO2_wn = wn[ NO2_idx ]
#     NO2_sp = sp[ NO2_idx ]

#     # Define the background
#     def line_between_two_points( x, x1, y1, x2, y2 ):
#         return y2 + ( y2 - y1 ) / ( x2 - x1 ) * ( x2 - x )
#     background_NO2_sp =  line_between_two_points( NO2_wn,  O3_baseline['x'], O3_baseline['y'], N2O_baseline['x'], N2O_baseline['y'] )
    
#     # NO2 integration
#     integral_NO2 = np.trapz( NO2_sp - background_NO2_sp, wn[ NO2_idx ] )

#     if( plot ):
#         plt.plot( draw_wn - draw_wn[0], draw_sp )
#         plt.plot( draw_wn - draw_wn[0], adv_plt.unp_n( background_NO2_sp ), '--' )
#         plt.fill_between( NO2_wn - draw_wn[0], adv_plt.unp_n( background_NO2_sp ), NO2_sp, alpha = 0.3, label = f"NO2: {integral_NO2:.4f}" )

#     return integral_NO2

def compute_NO2( wn, sp, plot = False, plot_with_real_wn = False ):
    # NO2-related area
    NO2_idx = ( wn > 1582 ) * ( wn < 1615 )
    NO2_wn = wn[ NO2_idx ]
    NO2_0wn = NO2_wn - NO2_wn[ 0 ]
    NO2_sp = sp[ NO2_idx ]

    # Detect if NO2 is up or down
    NO2_central_idx = np.max( np.where( wn < 1600 )[0] )
    NO2_a, _, _ = np.polyfit( wn[ NO2_central_idx - 4 : NO2_central_idx + 4 ], sp[ NO2_central_idx - 4 : NO2_central_idx + 4 ], 2 )
    NO2_direction = -1 if NO2_a > 0 else +1

    # Fitting function
    NO2_fitfun = lambda x, m, q, A1, s1, : A1 * np.exp( - ( x - 1597 + NO2_wn[ 0 ] )**2 / s1 ) + m * x + q
    p0 = [ ( NO2_sp[-1] - NO2_sp[0] ) / ( NO2_wn[-1] - NO2_wn[0] ), NO2_sp[0], np.ptp( NO2_sp ) * NO2_direction, 100 ]
    bounds = (
        # m,       q,       A1,      s1,   
        [ -np.inf, -np.inf, -np.inf,  0    ],
        [  np.inf,  np.inf,  np.inf,  1000 ]
    )
    
    try:
        p, pcov = curve_fit( NO2_fitfun, NO2_0wn, NO2_sp, p0, bounds=bounds )
    except (RuntimeError, RuntimeWarning):
        p = p0
        p[2] = 0
        pcov = np.zeros( ( len( p0 ), len( p0 ) ) )

    ( m, q ) = correlated_values( p[0:2], pcov[0:2,0:2] )

    # NO2-integrating area
    NO2_int_idx = ( wn > 1590 ) * ( wn < 1610 )
    NO2_int_wn = wn[ NO2_int_idx ] - NO2_wn[ 0 ]
    integral = np.trapz( sp[ NO2_int_idx ] - m * NO2_int_wn - q,  wn[ NO2_int_idx ] )

    fit_info = { 'm': p[0], 'x': np.mean( NO2_int_wn ) + NO2_wn[ 0 ], 'y': m * np.mean( NO2_int_wn ) + q }

    if( plot ):
        plt.plot( NO2_0wn, NO2_sp )
        plt.plot( NO2_0wn, NO2_fitfun( NO2_0wn, *p ) )
        plt.plot( NO2_0wn, NO2_fitfun( NO2_0wn, p[0], p[1], 0, p[3] ) )
        plt.fill_between( NO2_int_wn, p[0] * NO2_int_wn + p[1], sp[ NO2_int_idx ], alpha = 0.3, label = f"NO2: {integral:.4f}" )

    if( plot_with_real_wn ):
        plt.plot( NO2_0wn + NO2_wn[ 0 ], NO2_sp )
        plt.fill_between( NO2_int_wn + NO2_wn[ 0 ], p[0] * NO2_int_wn + p[1], sp[ NO2_int_idx ], alpha = 0.3, label = f"NO2: {integral:.4f}" )
        plt.plot( NO2_0wn + NO2_wn[ 0 ], NO2_fitfun( NO2_0wn, *p ) )

    return integral, fit_info


def compute_H2O2( wn, sp, plot = False ):
    # H2O2-related area
    H2O2_idx = ( wn > 1220 ) * ( wn < 1282 )
    H2O2_wn = wn[ H2O2_idx ]
    H2O2_0wn = H2O2_wn - H2O2_wn[ 0 ]
    H2O2_sp = sp[ H2O2_idx ]

    # Detect if H2O2 is up or down
    H2O2_central_idx = np.max( np.where( wn < 1272 )[0] )
    H2O2_a, _, _ = np.polyfit( wn[ H2O2_central_idx - 4 : H2O2_central_idx + 4 ], sp[ H2O2_central_idx - 4 : H2O2_central_idx + 4 ], 2 )
    H2O2_direction = -1 if H2O2_a > 0 else +1

    # Fitting function
    H2O2_fitfun = lambda x, m, q, A1, s1, : A1 * np.exp( - ( x - 1272 + H2O2_wn[ 0 ] )**2 / s1 ) + m * x + q
    p0 = [ ( H2O2_sp[-1] - H2O2_sp[0] ) / ( H2O2_wn[-1] - H2O2_wn[0] ), H2O2_sp[0], np.ptp( H2O2_sp ) * H2O2_direction, 100 ]
    bounds = (
        # m,       q,       A1,      s1,   
        [ -np.inf, -np.inf, -np.inf,  0    ],
        [  np.inf,  np.inf,  np.inf,  1000 ]
    )
    
    try:
        p, pcov = curve_fit( H2O2_fitfun, H2O2_0wn, H2O2_sp, p0, bounds=bounds )
    except (RuntimeError, RuntimeWarning):
        p = p0
        p[2] = 0
        pcov = np.zeros( ( len( p0 ), len( p0 ) ) )

    ( m, q ) = correlated_values( p[0:2], pcov[0:2,0:2] )

    # H2O2-integrating area
    H2O2_int_idx = ( wn > 1262 ) * ( wn < 1280 )
    H2O2_int_wn = wn[ H2O2_int_idx ] - H2O2_wn[ 0 ]
    integral = np.trapz( sp[ H2O2_int_idx ] - m * H2O2_int_wn - q,  wn[ H2O2_int_idx ] )

    fit_info = { 'm': p[0], 'x': np.mean( H2O2_int_wn ) + H2O2_wn[ 0 ], 'y': m * np.mean( H2O2_int_wn ) + q }

    if( plot ):
        plt.plot( H2O2_0wn, H2O2_sp )
        plt.plot( H2O2_0wn, H2O2_fitfun( H2O2_0wn, *p ) )
        plt.plot( H2O2_0wn, H2O2_fitfun( H2O2_0wn, p[0], p[1], 0, p[3] ) )
        plt.fill_between( H2O2_int_wn, p[0] * H2O2_int_wn + p[1], sp[ H2O2_int_idx ], alpha = 0.3, label = f"H2O2: {integral:.4f}" )

    return integral, fit_info

gases_labels = { 'O3': '$O_3$', 'N2O': '$N_2O$', 'NO2': '$NO_2$', 'H2O2': '$H_2O_2$' }

def estimate_areas( wn, spectra, quiet = False, plot = False ):
    
    if( spectra.ndim == 1 ):
        spectra = np.array( [ spectra ] )

    areas = { 'O3': []     , 'N2O': []      ,'NO2': [], 'H2O2': []}
    if( isinstance( plot, bool ) ):
        plot = list( areas.keys() ) if plot == True else []

    flip = False
    if( wn[-1] < wn[0] ):
        wn = np.flip( wn )
        flip = True
    
    for i, sp in ( enumerate( spectra ) if quiet else tqdm( list( enumerate( spectra ) ), leave = False ) ):
        sp = spectra[i,:]

        if( flip ):
            sp = np.flip( sp )

        sp = savgol_filter( sp, 13, 1 )

        O3, O3_baseline = compute_ozone( wn, sp, 'O3' in plot )
        areas['O3'].append( O3 )

        N2O, N2O_baseline = compute_N2O( wn, sp, 'N2O' in plot )
        # areas['CO2'].append( CO2 )
        areas['N2O'].append( N2O )
        
        # NO2 = compute_NO2_midpoint_interpolation( wn, sp, O3_baseline, N2O_baseline, 'NO2' in plot )
        NO2, NO2_baseline = compute_NO2( wn, sp, 'NO2' in plot )
        areas['NO2'].append( NO2 )

        H2O2, H2O2_baseline = compute_H2O2( wn, sp, 'H2O2' in plot )
        areas['H2O2'].append( H2O2 )

        if( plot ):
            plt.legend()
            plt.title( i )

        # if( i > 10 ):
            # break
    
    return areas