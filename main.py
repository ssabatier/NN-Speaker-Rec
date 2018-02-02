import os
import glob
import sys
from Preprocessing_Fn import *
import multiprocessing
import random
# import dircache
import sys
import os
import subprocess
import numpy as np
# from PyQt4.QtGui import *
# from PySide import QtGui, QtCore

"""
This script does the mean subtraction and normalization on the data.
The procedure is as follows:

    1 - The previously generated files(cubes which created per data block(which can be 1sec slots)) will be loaded.
        At this phase only files associated with the training data will be loaded. The mean and std will be calculated
        per channel. So for each channel we will have a mean and std image. The reason is that for example for channel[0]
        which contains static features, the mean is calculated based upon averaging over all features in different samples.
        So if each sample dimension is (3,40,98), then for channel[0] , the mean-image dimension is (40,98). Same applies
        for std-image calculation.

    2 - Now by feeding single by single file, the training and test data will be transformed to the new space.
        * A model should be applied on unseen data which the assumption is that it is not available at the test time.
          So in order to get a practical estimate of the test data, the calculation of the normalization parameters
          (mean and variance) should be restricted to the training set.

This file calls the following:

      Preprocessing_Scikit_Train: The function to get the 4D blob of training data and return the mean and variance.
      File_Transfer_Fn: The function which transform each single file to the new domain.
"""

"""
GUI Class definition
"""


# class MyButtons(QtGui.QDialog):
#     def __init__(self, choices, title):
#         # Initialized and super call.
#         super(MyButtons, self).__init__()
#         self.initUI(choices, title)
#         self.choice = choices
#
#     def initUI(self, choices, title):
#         option1Button = QtGui.QPushButton(choices[0])
#         option1Button.clicked.connect(self.onOption1)
#         option2Button = QtGui.QPushButton(choices[1])
#         option2Button.clicked.connect(self.onOption2)
#         option3Button = QtGui.QPushButton(choices[2])
#         option3Button.clicked.connect(self.onOption3)
#         option4Button = QtGui.QPushButton(choices[3])
#         option4Button.clicked.connect(self.onOption4)
#
#         buttonBox = QtGui.QDialogButtonBox()
#         buttonBox = QtGui.QDialogButtonBox(QtCore.Qt.Horizontal)
#         buttonBox.addButton(option1Button, QtGui.QDialogButtonBox.ActionRole)
#         buttonBox.addButton(option2Button, QtGui.QDialogButtonBox.ActionRole)
#         buttonBox.addButton(option3Button, QtGui.QDialogButtonBox.ActionRole)
#         buttonBox.addButton(option4Button, QtGui.QDialogButtonBox.ActionRole)
#         #
#         mainLayout = QtGui.QVBoxLayout()
#         mainLayout.addWidget(buttonBox)
#
#         self.setLayout(mainLayout)
#         # define window		xLoc,yLoc,xDim,yDim
#         self.setGeometry(250, 250, 100, 100)
#         self.setWindowTitle(title)
#         self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
#
#     def onOption1(self):
#         self.retStatus = 1
#         self.close()
#         self.choice = self.choice[0]
#
#     def onOption2(self):
#         self.retStatus = 2
#         self.close()
#         self.choice = self.choice[1]
#
#     def onOption3(self):
#         self.retStatus = 3
#         self.close()
#         self.choice = self.choice[2]
#
#     def onOption4(self):
#         self.retStatus = 4
#         self.close()
#         self.choice = self.choice[3]
#
#
# """
# GUI for training or testing phase.
# """
# app = QtGui.QApplication(sys.argv)
# user_options = ['TRAIN', 'TEST', 'Cancel', 'Continue']
# task_title = 'Are you intended to create testing or training pairs?!'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
choice_phase = 'TRAIN'
#
# # If user canceled the operation.
# if choice_phase == 'Cancel':
#     sys.exit("Canceled by the user")
#
# """
# GUI for getting the type of features.
# """
# user_options = ['logfbank_energy', 'fbank_energy', 'MFCC', 'raw']
# task_title = 'From which kind of features you want to create pairs?!'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
choice_feature = 'MFEC'

# Source and destination paths(both with absolute path).
this_path = os.path.dirname(os.path.abspath(__file__))
src_origin = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/2-CreateCubes/'
path_preprocessing_file = src_origin + 'Preprocessing_Statistics'
if not os.path.exists(path_preprocessing_file):
    os.makedirs(path_preprocessing_file)

src_folder_path = src_origin + 'CUBES/' + choice_feature + '/' + choice_phase
dst_folder_path = 'PreprocessedTrainTestData/' + choice_feature + '/' + choice_phase

# Get the list of all folders in the source directory
folders = [f for f in os.listdir(src_folder_path) if os.path.isdir(os.path.join(src_folder_path, f))]
num_folders = (
    len([name for name in os.listdir(src_folder_path) if os.path.isdir(os.path.join(src_folder_path, name))]))
print("Number of folders = ", num_folders)

# Get the number of all files in subdirectories
# The simplest way to get the output of a command is to use the subprocess.check_output function.
# However the output must be considered because it is the exact output of the terminal which
# might be of type "string" and we may want "int". Use prong function for tracking.
# "-type f" is necessary for just returning the number of files and do not count the folders.
num_files = subprocess.check_output('find ' + src_folder_path + ' -type f | wc -l', shell=True)

# subprocess.check_output returns the number of files as '1518\n'. The last element '\n' must be
# eliminated and it count only 1 character in python.
num_files = int(num_files[:-1])

# Print the number of files.
print("Number of files", num_files)

# Read a random file for getting the shapes of the files.
# RandFolder = random.choice(dircache.listdir(src_folder_path))
RandFolder = folders[1]


FileShape = np.load(glob.glob(os.path.join(src_folder_path, RandFolder, '*.npy'))[0]).shape
# FileShape = (3,40,98)
# FileShape = FileShape.reshape(numpy_array.shape[2],numpy_array.shape[1],numpy_array.shape[0])
print("File shape: ", FileShape)

# This part is specific for the feature cube of pairs that have been generated.
N = num_files
X = np.zeros((N, FileShape[0], FileShape[1],FileShape[2]),
             dtype=np.float32)  # the number 2 is because the hog features of a pairs should be considered separately
y = np.zeros(N, dtype=np.int64)  #
FileNum = np.zeros(N, dtype=np.int64)

# Creating random number for shuffling the files.
Rand_idx = np.random.permutation(range(N))

# Looping over all folders and files.
# All the data is fed to the big X vector that has been created.
# The shuffling of the data is done here before creating LMDB file.
# Class labels are defined based on distinct folder names.

if choice_phase == 'TRAIN':
    print("Calculation of mean and std arrays for the training data has just started!")
    counter = 0
    for subject_class, folder in enumerate(folders):
        for file_name in glob.glob(os.path.join(src_folder_path, folder, "*.npy")):

            numpy_array = np.load(file_name)
            # numpy_array = numpy_array.reshape(numpy_array.shape[2],numpy_array.shape[1],numpy_array.shape[0])

            # Feed the data cube.
            X[Rand_idx[counter], :, :, :] = numpy_array

            # Different labels are assigned for different subfolders(IDs).
            y[Rand_idx[counter]] = subject_class

            if counter % 100 == 0:
                print("Processing %d file" % counter)
            counter = counter + 1

    # Create a big array in order to turn into LMDB
    X = np.delete(X, np.s_[counter:X.shape[0]], 0)  # delete extra pre-allocated space

    # Pre-processing on the training data
    # The transformation is done only if it is necessary because it is time-consuming.
    mean_image, std_image, X_train = Preprocessing_Scikit_Train(X, transform_status=False)

    np.save(os.path.join(path_preprocessing_file, 'mean_image'),
            mean_image)
    np.save(os.path.join(path_preprocessing_file, 'std_image'),
            std_image)

print("Loading mean and std files...")
mean_image = np.load(os.path.join(path_preprocessing_file, 'mean_image.npy'))
std_image = np.load(os.path.join(path_preprocessing_file, 'std_image.npy'))

print("Data is being transformed...")

for subject_class, folder in enumerate(folders):
    print("Processing folder number %d of %d" % (subject_class + 1, num_folders))
    ID_Folder = dst_folder_path + '/' + folder
    if not os.path.exists(ID_Folder):
        os.makedirs(ID_Folder)

    for file_name in glob.glob(os.path.join(src_folder_path, folder, "*.npy")):
        numpy_array = np.load(file_name)

        # Feed the data cube.
        feature_cube = File_Transfer_Fn(numpy_array, mean_image, std_image)
        np.save(
            os.path.join(ID_Folder, 'preprocessed' + '_' + os.path.basename(file_name).split('.')[0]),
            feature_cube)
