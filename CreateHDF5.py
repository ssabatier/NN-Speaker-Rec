import os
import glob
import sys
import multiprocessing
import random
# import dircache
import numpy as np
# from PyQt4.QtGui import *
# from PySide import QtGui, QtCore
import h5py
from os import listdir
from os.path import isfile, join

"""
This code reads all the numpy files and create HDF5 file with TensorFlow format(num_samples,height,width,channel).
Saved data format is (channels,height,width) which is adaptable with Caffe.
For being adaptable by TensorFlow the data format has to be transformed to (height,width,channel)
"""
choice_phase = 'TEST'

# # If user canceled the operation
# if choice_phase == 'Cancel':
#     sys.exit("Canceled by the user")
choice_feature = 'raw'

# Source and destination paths(both with absolute path)
this_path = os.path.dirname(os.path.abspath(__file__))
src_folder_path = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/4-CreatePairs/PAIRS/' + choice_feature + '/' + choice_phase
dst_folder_path = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/5-CreateData/' + choice_feature + '_' + choice_phase + '_' + 'HDF5'

# Getting the number of files in the folder
num_files = (len([name for name in os.listdir(src_folder_path) if os.path.isfile(os.path.join(src_folder_path, name))]))
print("Number of files = ", num_files)

# Read a random file for getting the shapes
# RandFile = random.choice(dircache.listdir(src_folder_path))
files = [f for f in listdir(src_folder_path) if isfile(join(src_folder_path,f))]
RandFile = files[1]
FileShape = np.load(os.path.join(src_folder_path, RandFile)).shape
print("File shape: ", FileShape)

# Pre-allocating space for HDF5 file
hdf5_file = h5py.File(choice_phase + '_' + choice_feature + '.hdf5', 'w')
hdf5_file.create_dataset("pairs", shape=(num_files, FileShape[1], FileShape[2], 2 * FileShape[0]),
                         dtype='float32')
hdf5_file.create_dataset("labels", shape=(num_files, 1),
                         dtype=np.int)

# Initialize a counter
counter = 0

# Shuffle the order to save the files in LMDB format
Rand_idx = np.random.permutation(range(num_files))

# Read all the numpy files
# Saved data format is (channels,height,width)
# To be compatible with tensorflow, the data format has to be transformed to (height,width,channel)
for f in glob.glob(os.path.join(src_folder_path, "*.npy")):
    # Load numpy file
    # format is (channels,height,width,2)
    numpy_pair = np.load(f)

    # Separate two parts of a pair.
    left = np.transpose(numpy_pair[:, :, :, 0],
                        (1, 2, 0))
    right = np.transpose(numpy_pair[:, :, :, 1],
                         (1, 2, 0))

    # Save to HDF5 file.
    hdf5_file["pairs"][Rand_idx[counter], :, :, 0: numpy_pair.shape[0]] = left
    hdf5_file["pairs"][Rand_idx[counter], :, :, numpy_pair.shape[0]:] = right

    # If the pairs are genuine, then it has "gen" in its name
    if 'gen' in f:
        hdf5_file["labels"][Rand_idx[counter]] = 1
    else:
        hdf5_file["labels"][Rand_idx[counter]] = 0
    if counter % 100 == 0:
        print("Processing %d pairs" % counter)
    counter = counter + 1
