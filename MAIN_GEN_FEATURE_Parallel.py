"""
This file is the main file for generating features using parallel computing.
This file does the following:
   1. Get the source folder's files
   2. Extract speech features with defined parameters
   4. Save the features using numpy format and with the same file name as original file

The functions that will be called within this file are:
    "gather_Gen_feature_data"
"""
# import h5py
import dask.array as da
import numpy as np
import sys
import scipy.io as sio
import glob
import os
from timeit import default_timer as timer
from Gen_Feature import Gen_Feature_Speech
# from easygui import *
# from joblib import Parallel, delayed
from openpyxl import load_workbook, Workbook
import multiprocessing
import shutil
import sklearn
# import scikits
# from PyQt4.QtGui import *
# from PySide import QtGui, QtCore
import speechpy

num_cores_max = multiprocessing.cpu_count()
num_cores = num_cores_max - 2

"""
GUI Class definition
"""
# class MyButtons(QtGui.QDialog):
#     """"""
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
# """
# GUI for training or testing phase.
# """
# app = QtGui.QApplication(sys.argv)
# user_options = ['YES', 'NO', 'Cancel', 'Continue']
# task_title = 'Do you want to stack frames?'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
# choice_stack = form.choice
choice_stack = 'NO'
# # If user canceled the operation.
#
# if choice_stack == 'Cancel':
#     sys.exit("Canceled by the user")
#
# """
# GUI for getting the type of features.
# """
# user_options = ['logfbank_energy', 'fbank_energy', 'MFCC', 'raw']
# task_title = 'With which features would you like to create pairs?'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
# choice_feature = form.choice
choice_feature = 'MFEC'
#
# 
# year = str(integerbox(msg='From which year would you like to extract features?', title='Extracting Features',
#                   default='2014', lowerbound=2014, upperbound=2015))
year = '2014'
"""
Part 2: Generating Pairs
"""
SRC_FOLDER = '/home/stallone/Desktop' + '/' + 'Twins'  + str(year)  + 'Data' + '/' + 'Audio'
DST_FOLDER = 'FEATURES' + '/' + choice_feature
# Make sure that the destination folder directory exists
# Creating the directory.
if not os.path.exists(DST_FOLDER):
    os.makedirs(DST_FOLDER)

# Load IDs
if year== '2014':
    # ID_2014 = np.load('REPORT/2014_Report.npy').astype(int)
    # # # ID_2014 = np.load('twinRelId14.npy').astype(int)
    # ID = ID_2014[:, 0]
    ID_list = os.listdir(SRC_FOLDER)
    print(ID_list)

elif year=='2015':
    # ID_2015 = np.load('REPORT/2015_Report.npy').astype(int)
    # ID_2015 = np.load('twinRelId15.npy').astype(int)
    # ID = ID_2015[:, 0]
    ID_list = os.listdir(SRC_FOLDER)
    print(ID_list)
elif year=='2016':
    # ID_2015 = np.load('REPORT/2015_Report.npy').astype(int)
    # ID_2016 = np.load('twinRelId16.npy').astype(int)
    # ID = ID_2016[:, 0]
    ID_list = os.listdir(SRC_FOLDER)
    print(ID_list)

# ID to list for processing
# ID_list = ID.tolist()

#### SESSIONS ####

if year == '2014':
    sessions = ['08022014', '08032014']
elif year == '2015':
    sessions = ['08082015', '08092015']
elif year == '2016':
    sessions = ['08062016', '08072016']

# file_names = SRC_FOLDER,str(ID),'08022014','Nuemann Mic','Audio_VAD','*VAD.wav'
# print(file_names)
# print('doo')
# Running with parallel computing
start = timer()
# Parallel(n_jobs=num_cores)(
#     delayed
for ID in ID_list:
    Gen_Feature_Speech(ID, SRC_FOLDER, DST_FOLDER, sessions, choice_stack, choice_feature)
