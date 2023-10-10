import numpy as np
import pandas as pd
from glob import glob
import re

def read_axy(filepath):
    df = pd.read_csv( filepath, index_col=False, names=["Time","Intensity","Dunno"] )

    return {
        't': df['Time'].to_numpy(),
        'I': df['Intensity'].to_numpy(),
        'I1': df['Dunno'].to_numpy(),
    }

def read_axy_series(path):
    
    data = []
    t = []
    ids = []

    pattern = ".*[_/\\\\](\d+)\.AXY"
    for g in glob( path + "/*.AXY" ):
        matches = re.match( pattern, g )
        if( not matches ): continue

        ids.append( matches.groups()[0] )
        d = read_axy(g )
        t = d['t']
        data.append( d['I'] )

    data = np.array( data )[ np.argsort( ids ) ]

    return {
        't': np.array( t ),
        'I': data
    }