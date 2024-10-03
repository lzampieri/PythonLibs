from standard_imports import *

def max_fromfit( ys, thresh = 0.1, max_outliners = 2, return_x = False ):
    st = ed = np.argmax( ys )
    lev = ys[st] - thresh * np.ptp( ys )

    outliners = 0
    while( outliners < max_outliners ):
        st -= 1
        if( ys[st] < lev ):
            outliners += 1
    outliners = 0
    while( outliners < max_outliners ):
        ed += 1
        if( ys[ed] < lev ):
            outliners += 1

    fit = np.polyfit( range( ed-st ), ys[st:ed], 2 )
    x_max = - fit[1] / 2 / fit[0]
    if( x_max < 0 or x_max > ed - st ):
        print( "Fit failed! Using average" )
        return np.mean( ys[ st - ed ] )
    
    # xs = range( ed-st )
    # plt.plot( xs, ys[st:ed], 'o' )
    # plt.plot( xs, np.polyval( fit, xs ))
    # plt.plot( [x_max], [np.polyval( fit, x_max )], 'o' )
    
    # print( st, x_max, ed )
    if( return_x ):
        return np.polyval( fit, x_max ), x_max + st
    return np.polyval( fit, x_max )


def min_fromfit( ys, thresh = 0.1, max_outliners = 2, return_x = False ):
    return -max_fromfit( - np.array(ys), thresh, max_outliners, return_x )

def ptp_fromsavgol( ys, polyorder = 3 ):
    if( polyorder % 2 == 0 ):
        polyorder = polyorder + 1

    data = [ np.ptp( savgol_filter( ys, i, polyorder ) ) for i in range( polyorder + 2, polyorder + 25, 2 ) ]

    return mean( data )
