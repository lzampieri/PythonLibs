

def propane_fake_flux_over_air( propane_fake_mlmin, air_lmin ):
    propane_real = propane_fake_mlmin * 0.34
    return propane_real / ( air_lmin * 1000 + propane_real ) * 1e6