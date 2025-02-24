import pandas as pd
import re

def load( filename ):

    # Check on the fourth line, to understand if it has params or not
    has_params = False
    with open(filename) as f:
        line = ""
        for i in range(4):
            line = f.readline()
        if( not re.match( r"^[+\-\deE,\.]+$", line) ):
            has_params = True # If the forth line it not only numers and dots, the file have an header!

    dataframe = pd.read_csv( filename, header=0, skiprows= ( list(range(10)) + [11] ) if has_params else [1], index_col=False )

    data = {}
    for column in dataframe.columns:
        data[ 'Time' if column == 'Source' else column ] = dataframe[ column ].to_numpy()

    return data
