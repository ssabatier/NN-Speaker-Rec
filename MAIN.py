import h5py
import dask.array as da
import numpy
import sys
import scipy.io as sio
import glob
import itertools
import os
# import cv2
# from easygui import integerbox
from Gen_Cube_Fn import Generate_Cube
# from joblib import Parallel, delayed
import multiprocessing
import shutil
# from PyQt4.QtGui import *
# from PySide import QtGui, QtCore

# Getting the number of cores: Necessary for parallel computing.
num_cores = multiprocessing.cpu_count()

"""
# GUI Class definition
# class MyButtons(QtGui.QDialog):
#     """"""
#
#     def __init__(self, choices, title):
#         # Initialized and super call
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
#         
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
# """

# GUI for training or testing phase.
# """
# app = QtGui.QApplication(sys.argv)
# user_options = ['train', 'test', 'Cancel', 'Continue']
# task_title = 'Do you wish to create testing or training pairs?'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
choice_phase = 'test'

# # If user canceled the operation.
# if choice_phase == 'Cancel':
#     sys.exit("Canceled by the user")
#
# """

# GUI for getting the type of features
# """
# user_options = ['logfbank_energy', 'fbank_energy', 'MFCC', 'raw']
# task_title = 'With which features would you like to make pairs?'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
choice_feature = 'raw'
# """

# GUI for getting the session
# """
# user_options = ['first', 'second', 'both', 'Cancel']
# task_title = 'From which session would you like to create pairs?'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
choice_session = 'both'
# user_options = ['rainbow', 'subject', 'both']
choice_file = 'rainbow'
# """

# GUI for getting the year for training.
# """
if choice_phase == 'train':
    TRAIN_YEARS = []
    user_options = ['2014', '2015', 'both', 'Cancel']
    # task_title = 'From which year would you like to create pairs for training?'
    # form = MyButtons(choices=user_options, title=task_title)
    # form.exec_()
    choice_year = 'both'
    if choice_year == '2014':
        TRAIN_YEARS.append(user_options[0])
    elif choice_year == '2015':
        TRAIN_YEARS.append(user_options[1])
    elif choice_year == 'both':
        TRAIN_YEARS.append(user_options[0])
        TRAIN_YEARS.append(user_options[1])
    else:
        sys.exit("Canceled by the user!")
    YEARS = TRAIN_YEARS
# """

# GUI for getting the year for testing.
# """
if choice_phase == 'test':
    TEST_YEARS = []
    user_options = ['2014', '2015', 'both', 'Cancel']
    # task_title = 'From which year would you like to create pairs for testing?!'
    # form = MyButtons(choices=user_options, title=task_title)
    # form.exec_()
    choice_year = '2015'
    if choice_year == '2014':
        TEST_YEARS.append(user_options[0])
    elif choice_year == '2015':
        TEST_YEARS.append(user_options[1])
    elif choice_year == 'both':
        TEST_YEARS.append(user_options[0])
        TEST_YEARS.append(user_options[1])
    else:
        sys.exit("Canceled by the user!")
    YEARS = TEST_YEARS

## Number of frames per each feature cube
# number_frames = integerbox(
#     msg='How many frames do you want to use?',
#     title='Extracting Features',
#     default=98, lowerbound=0, upperbound=1000)

# number_frames is the number of frames or samples (for raw data) that we want cut to make cubes
# Ex. For 16k sampling rate, number_frames = 6400 corresponds to 400 ms chunks of data
number_frames = 6400

# If the stride is equal to number_frames, then there is no overlap.
# overlap_stride = integerbox(
#     msg='What is the stride?',
#     title='Extracting Features',
#     default=number_frames, lowerbound=0, upperbound=1000)
overlap_stride = 6400

"""
Part 2: Generating Pairs
"""
SRC_FOLDER = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/1-GenerateFeatures/FEATURES'
dst_origin = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/2-CreateCubes/'
DST_FOLDER_TRAIN = dst_origin + 'CUBES/' + choice_feature + '/TRAIN'
DST_FOLDER_TEST = dst_origin + 'CUBES/' + choice_feature + '/TEST'

"""
GUI for removing the files that have been generated previously.
"""
# a = QApplication(sys.argv)
# # The QWidget widget is the base class of all user interface objects in PyQt4.
# w = QWidget()
# # Show a message box
# result = QMessageBox.question(w, 'Action required', "Do you want to remove previously generated files?",
#                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
result = 'Yes'

if result == 'Yes':
    remove_status = True
else:
    remove_status = False
# Show window
# w.show()

if choice_phase == 'train':
    # Remove previous files
    if remove_status:
        if os.path.exists(DST_FOLDER_TRAIN):
            os.system("rm -rf %s" % DST_FOLDER_TRAIN)

    # Create the directory.
    if not os.path.exists(DST_FOLDER_TRAIN):
        os.makedirs(DST_FOLDER_TRAIN)
else:
    # Remove previous files
    if remove_status:
        if os.path.exists(DST_FOLDER_TEST):
            os.system("rm -rf %s" % DST_FOLDER_TEST)

    # Create the directory.
    if not os.path.exists(DST_FOLDER_TEST):
        os.makedirs(DST_FOLDER_TEST)

# Load metadata
ID_2014_RELATIONS = numpy.load('/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/0-VAD/twinRelId14.npy').astype(int)
ID_2015_RELATIONS = numpy.load('/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/0-VAD/twinRelId15.npy').astype(int)
relations = numpy.concatenate((ID_2014_RELATIONS, ID_2015_RELATIONS), axis=0)

# Get the IDs
ID_2014_L = ID_2014_RELATIONS[:, 0].reshape(ID_2014_RELATIONS.shape[0], 1)
ID_2014_R = ID_2014_RELATIONS[:, 1].reshape(ID_2014_RELATIONS.shape[0], 1)
ID_2014 = numpy.concatenate((ID_2014_L,ID_2014_R),axis=0)
ID_2015_L = numpy.unique(ID_2015_RELATIONS[:, 0].reshape(ID_2015_RELATIONS.shape[0], 1))
ID_2015_R = numpy.unique(ID_2015_RELATIONS[:, 1].reshape(ID_2015_RELATIONS.shape[0], 1))
ID_2015 = numpy.concatenate((ID_2015_L,ID_2015_R),axis=0)

# Get the ids for training .
# ID_2015_seen = numpy.intersect1d(ID_2014, ID_2015)
# ID_2014_2015 = numpy.concatenate((ID_2014, ID_2015_seen),axis=0)
# ID_2014_2015 = numpy.unique(ID_2014_2015)
train_id = ID_2014.astype(int)[:, 0]
train_id_list = train_id.tolist()  # Necessary for parallel computing

# Get the ids for testing.
test_id_2015_unseen = numpy.setdiff1d(ID_2015, numpy.intersect1d(ID_2014, ID_2015)).astype(int)
test_id_2015_seen = numpy.intersect1d(ID_2014, ID_2015)
test_id = test_id_2015_unseen.astype(int)
test_id_list = test_id.tolist()
# un_2014 = numpy.unique(ID_2014)
# un_2015 = numpy.unique(ID_2015)
# un_2015L = numpy.unique(ID_2015_L)
# un_2015R = numpy.unique(ID_2015_R)
# id_2015_overlap = numpy.intersect1d(ID_2015_L, ID_2015_R)

"""
Get the sessions and return them as a list.
"""
session_initial = []
sessions = ['08022014', '08082015', '08032014', '08092015']
kinds = ['rainbow','0004']

for session in sessions:
    if any(year in session for year in YEARS):
        session_initial.append(session)

session_list = []
if choice_session == 'first':
    session_list.append(session_initial[0])

elif choice_session == 'second':
    session_list.append(session_initial[1])

elif choice_session == 'both':
    for i in range(len(session_initial)):
        session_list.append(session_initial[i])

else:
    sys.exit("Cancelled by the user at line 256!")

# Call multiple processors for creating pairs
if choice_phase == 'test':
    for ID in test_id_list:
        Generate_Cube(ID, SRC_FOLDER, DST_FOLDER_TEST, YEARS,
                             choice_phase, choice_feature, session_list, number_frames, overlap_stride)

else:
    for ID in train_id_list:
        Generate_Cube(ID, SRC_FOLDER, DST_FOLDER_TRAIN, YEARS,
                             choice_phase, choice_feature, session_list, number_frames, overlap_stride)
