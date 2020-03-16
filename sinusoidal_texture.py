"""
defines help functions for constructing a sinusoidal texture for meshing
a layer interface with the finite element solver JCMsuite.
"""

import numpy as np


def get_texture(pitch,aspect_ratio,phase):
    '''
    returns the string holding a python evaluation for the texture height Z at
    each position in the x-y plane. Positions are evaluated by the tuple X,
    X[0] is the x position and X[1] is the y position.
    '''
    factor = np.pi * 2/(np.sqrt(3) * pitch)
    phase = np.pi*phase
    h_peak_valley = get_peak_valley(phase)    
    amplitude = pitch*aspect_ratio
    ampl = amplitude / h_peak_valley    
    string = "_ampl_ * (cos(X[0] * _factor_+_phase_) * " + \
             "cos(0.5 * _factor_ * (X[0] + sqrt(3.) * X[1])) * " + \
             "cos(0.5 * _factor_ * (X[0] - sqrt(3.) * X[1])))"
    string = string.replace('_factor_', str(factor))
    string = string.replace('_ampl_', str(ampl))
    string = string.replace('_phase_', str(phase))
    return string

def get_peak_valley(phase):
    phi = np.abs(phase)
    return 1.29903810568*np.sin(phi/3.+np.pi/3.)
    
