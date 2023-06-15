import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat

def n( x ):
    return x.n if 'uncertainties' in str( type( x ) ) else x
def s( x ):
    return x.s if 'uncertainties' in str( type( x ) ) else 0