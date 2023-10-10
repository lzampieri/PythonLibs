import numpy as np
import pandas as pd

def read_evt(filepath):
    df = pd.read_excel( filepath, index_col=False ).to_dict('records')

    return df