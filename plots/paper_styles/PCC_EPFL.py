from standard_imports import *
from plots.splt import *

def style():
    set_export_folder( "D:\DriveUniMiB\A_Articles\PCC_EPFL\img" )
    enable_seaborn()
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("white", { 'axes.grid': True })


style()

def Vaxis():
    plt.xlabel("Peak voltage [kV]")

def Daxis():
    plt.xlabel("Nozzle-water surface distance [cm]")