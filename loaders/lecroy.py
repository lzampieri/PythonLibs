import pandas as pd

def load( filename ):
    data = pd.read_csv( filename, header=4 )

    return {
        'Time': data['Time'].to_numpy(),
        'Ampl': data['Ampl'].to_numpy()
    }
