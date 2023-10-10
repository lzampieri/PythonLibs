import numpy as np
from scipy import ndimage
from uncertainties.core import Variable
from uncertainties import ufloat
from standard_imports import mean





def realign_couple( raw_plasma_off, raw_plasma_on, wave_info_plasma_off, wave_info_plasma_on ):

    # Define the alignments
    phase_on_wrt_off =  wave_info_plasma_on[ 'hv' ][ 'phase_t' ] - wave_info_plasma_off[ 'hv' ][ 'phase_t' ]
    phase_rog_wrt_hv = wave_info_plasma_off[ 'rog' ][ 'phase_t' ] - wave_info_plasma_off[ 'hv' ][ 'phase_t' ]

    # Compute the shifts
    time_step = time_step_computer( raw_plasma_off[ 'hv' ] )
    shift_hv_off  = ( ufloat( 0, 0 ) ) / time_step
    shift_hv_on   = ( phase_on_wrt_off ) / time_step
    shift_rog_off = ( phase_rog_wrt_hv ) / time_step
    shift_rog_on  = ( phase_rog_wrt_hv + phase_on_wrt_off ) / time_step

    # Compute the limits to avoid boundaries effects
    min_idx =                                       int( np.max( [ 0, -shift_hv_off.n, -shift_hv_on.n, -shift_rog_off.n, -shift_rog_on.n ]  ) )
    max_idx = len( raw_plasma_off['hv']['Ampl'] ) - int( np.max( [ 0,  shift_hv_off.n,  shift_hv_on.n,  shift_rog_off.n,  shift_rog_on.n ]  ) )

    # Reduce the limits such that they're a integer number of periods
    period_steps = int( wave_info_plasma_off['hv'][ 'T' ].n / time_step )
    n_periods = 1
    while( period_steps * ( n_periods + 1 ) < max_idx - min_idx ):
        n_periods = n_periods + 1
    min_idx = int( ( min_idx + max_idx ) / 2 - period_steps * n_periods / 2 )
    max_idx = min_idx + period_steps * n_periods

    # Shift!
    realigned_plasma_off = {
        'hv':  shift_waveform( raw_plasma_off['hv']['Ampl'] , shift_hv_off  ) [ min_idx : max_idx ],
        'rog': shift_waveform( raw_plasma_off['rog']['Ampl'], shift_rog_off ) [ min_idx : max_idx ],
    }
    realigned_plasma_on = {
        'hv':  shift_waveform( raw_plasma_on['hv']['Ampl'] , shift_hv_on  ) [ min_idx : max_idx ],
        'rog': shift_waveform( raw_plasma_on['rog']['Ampl'], shift_rog_on )[ min_idx : max_idx ]
    }

    # Compute the residual phase
    residual_phase = wave_info_plasma_on[ 'rog' ][ 'phase_t' ] - wave_info_plasma_on[ 'hv' ][ 'phase_t' ] - phase_rog_wrt_hv

    return {
        'plasma_off': realigned_plasma_off,
        'plasma_on' : realigned_plasma_on,
        }, {
        'plasma_off': ufloat( 0, 0 ),
        'plasma_on' : residual_phase,
        }, n_periods
