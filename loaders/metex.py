import numpy as np
import pandas as pd

def load(filepath):
    df = pd.read_csv( filepath, names=['timestamp', 'value', 'info'], index_col=False )
    return { k: df[k].to_numpy() for k in df.columns }