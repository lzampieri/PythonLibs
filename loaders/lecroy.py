import pandas as pd

def load( filename ):
    data = pd.read_csv( filename, header=4 )

    return data
