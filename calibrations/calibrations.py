import numpy as np
import re
from datetime import datetime
from os.path import dirname, exists
from uncertainties import ufloat

def save(name, values, note = None ):
    name = name.lower()
    assert re.match( "^[a-z0-9_]+$", name ), "Only letters, numbers and underscores are allowed"

    filename = "c" + name + ".cals"

    descr = {
        'name': name,
        'saved_on': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    }
    if( note ):
        descr['note'] = note

    with open( dirname( __file__ ) + "/" + filename, 'w', encoding="utf-8") as file:
        for k in descr.keys():
            file.write( str(k) + "|" + str( descr[k] ) + "\n" )
    
        file.write( "values|values\n" )
        for v in values:
            if 'uncertainties' in str( type( v ) ):
                file.write( str(v.n) + "|" + str(v.s) + "\n" )
            else:
                file.write( str(v) + "\n" )
        file.flush()

    print( "Saved! -", filename )

def calibrate(name, data ):
    name = name.lower()
    assert re.match( "^[a-z0-9_]+$", name ), "Only letters, numbers and underscores are allowed"

    filename = "c" + name + ".cals"

    if not exists( dirname( __file__ ) + "/" + filename ):
        if exists( dirname( __file__ ) + "/" + "c" + name + ".calib" ):
            assert False, "The calibration file " + "c" + name + ".calib" + " is only available in a older version no more supported. Please update."
        assert False, "File " + filename + " not found!"

    descr = {}
    values = []
    on_values = False

    with open( dirname( __file__ ) + "/" + filename, 'r', encoding="utf-8") as file:
        for line in file:
            clean_line = line.rstrip()
            if( not on_values ):
                if( clean_line.find( "|" ) != -1 ):
                    k, v = clean_line.split( "|", 1 )
                    if( k == 'values' and v == 'values' ):
                        on_values = True
                        continue
                    descr[ k ] = v
            else:
                if( line.find( "|" ) != -1 ):
                    n, s = clean_line.split( "|", 1 )
                    values.append( ufloat( float( n ), float( s ) ) )
                else:
                    values.append( float( clean_line ) )

    print( "Calibrating using file " + filename + " generated on " + descr['saved_on'] if 'saved_on' in descr.keys() else "??" )
    if( 'note' in descr.keys() ):
        print( "Note: ", descr['note'] )

    return np.polyval( values, data )