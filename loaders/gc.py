import numpy as np
import pandas as pd

def read_axy(filepath):
    df = pd.read_csv( filepath, index_col=False, names=["Time","Intensity","Dunno"] )

    return {
        't': df['Time'].to_numpy(),
        'I': df['Intensity'].to_numpy(),
        'I1': df['Dunno'].to_numpy(),
    }