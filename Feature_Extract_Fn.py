import scipy.io as sio
import numpy as np
import os
import sys
from SPEECH_SIGNAL_PROCESSING import mfcc
from SPEECH_SIGNAL_PROCESSING import fbank
from SPEECH_SIGNAL_PROCESSING import logfbank

def ExtractFeature(signal,rate, feature_type):
    """ This part is for HOG feature extraction******

        If the choice_cube = True:
             then the software creates a HOG cube based on the
             vlfeat package which generate(?,?,31) dimensional HoG.
             # Note: The default cell size in 8.
        Otherwise:
             the scikit_image package will be used.
    """

    """ This part is for detecting and cropping the face"""
    if feature_type == 'fbank_energy':
        out_signal = fbank(signal, rate)

    elif feature_type == 'logfbank_energy':
        out_signal = logfbank(signal, rate)

    elif feature_type == 'MFCC':
        out_signal = mfcc(signal, rate)

    else:
        out_signal = signal

    return out_signal