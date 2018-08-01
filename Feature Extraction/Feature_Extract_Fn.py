import scipy.io as sio
import numpy as np
import os
import sys
from SPEECH_SIGNAL_PROCESSING import mfcc
from SPEECH_SIGNAL_PROCESSING import fbank
from SPEECH_SIGNAL_PROCESSING import logfbank

def ExtractFeature(signal,rate, feature_type):
    
    if feature_type == 'fbank_energy':
        out_signal = fbank(signal, rate)

    elif feature_type == 'logfbank_energy':
        out_signal = logfbank(signal, rate)

    elif feature_type == 'MFCC':
        out_signal = mfcc(signal, rate)

    else:
        out_signal = signal

    return out_signal
