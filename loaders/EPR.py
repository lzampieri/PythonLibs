import pandas as pd

def load( filename ):
    dataframe = pd.read_csv( filename, delim_whitespace=True, names = ['Field', 'Intensity'], index_col=0, skiprows=[0])

    return { 'Field': dataframe['Field'].to_numpy(), 'Intensity': dataframe['Intensity'].to_numpy(), 'filename': filename }
