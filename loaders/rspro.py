import pandas as pd

def load( filename ):
    dataframe = pd.read_csv( filename, header=0, skiprows=[1], index_col=False )

    data = {}
    for column in dataframe.columns:
        data[ 'Time' if column == 'Source' else column ] = dataframe[ column ].to_numpy()

    return data
