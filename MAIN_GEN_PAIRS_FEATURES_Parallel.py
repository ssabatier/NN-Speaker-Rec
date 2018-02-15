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
from Gen_Genuine_pairs_Fn_Parallel import Generate_Genuine
from Gen_Impostor_pairs_Fn_Parallel import Generate_Impostor
# from joblib import Parallel, delayed
import multiprocessing
import shutil
# from PyQt4.QtGui import *
# from PySide import QtGui, QtCore

# Getting the number of cores: Necessary for parallel computing.
num_cores = multiprocessing.cpu_count()

# """
# GUI for training or testing phase.
# """
# app = QtGui.QApplication(sys.argv)
# user_options = ['TRAIN', 'TEST', 'Cancel', 'Continue']
# task_title = 'Are you intended to create testing or training pairs?!'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
# choice_phase = form.choice
choice_phase = 'TEST'

# # If user canceled the operation.
# if choice_phase == 'Cancel':
#     sys.exit("Canceled by the user")
#
# """
# GUI for getting the type of features.
# """
# user_options = ['logfbank_energy', 'MFEC', 'MFCC', 'raw']
# task_title = 'From which kind of features you want to create pairs?!'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
# choice_feature = form.choice
choice_feature = 'raw'

# """
# GUI for getting the session.
# """
# user_options = ['Speech_Aligned', 'Speech_Not_Aligned', 'IDC', 'Cancel']
# task_title = 'How do you want to create genuine pairs?!'
# form = MyButtons(choices=user_options, title=task_title)
# form.exec_()
# choice = form.choice
# if choice == 'Speech_Aligned':
#     speech_aligned = True
#     print "speech_aligned corresponds to ASR"
# elif choice == 'Speech_Not_Aligned':
#     speech_aligned = False
#     print "Speech_Not_Aligned corresponds to SRE"
# else:
#     sys.exit('cancelled by the user')
speech_aligned = False


# # Number of frames per each feature cube
# number_frames = integerbox(
#     msg='What number of frames you want to use(it is determined base on the frame length and overlap)?',
#     title='Extracting Features',
#     default=98, lowerbound=0, upperbound=1000)


"""
Part 2: Generating Pairs
"""
SRC_FOLDER = '/home/stallone/Documents/PyCharm_Scripts/speech_processing-master/3-preprocessing/PreprocessedTrainTestData/' + choice_feature + '/' + choice_phase
DST_FOLDER_TRAIN = 'PAIRS/' + choice_feature + '/TRAIN'
DST_FOLDER_TEST = 'PAIRS/' + choice_feature + '/TEST'

"""
GUI for removing the files that been generated previously.
"""
#
# a = QApplication(sys.argv)
# # The QWidget widget is the base class of all user interface objects in PyQt4.
# w = QWidget()
# # Show a message box
# result = QMessageBox.question(w, 'Action required', "Do you want to remove previously generated files?",
#                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
result = 'yes'
if result == 'yes':
    remove_status = True
else:
    remove_status = False
# # Show window
# w.show()

if choice_phase == 'TRAIN':
    # Remove previous files
    if remove_status:
        if os.path.exists(DST_FOLDER_TRAIN):
            os.system("rm -rf %s" % DST_FOLDER_TRAIN)

    # Creating the directory.
    if not os.path.exists(DST_FOLDER_TRAIN):
        os.makedirs(DST_FOLDER_TRAIN)
else:
    # Remove previous files
    if remove_status:
        if os.path.exists(DST_FOLDER_TEST):
            os.system("rm -rf %s" % DST_FOLDER_TEST)

    # Creating the directory.
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
# ID_2014_2015 = numpy.vstack((ID_2014, ID_2015))
# ID_2014_2015 = numpy.unique(ID_2014_2015)
train_id_gen = ID_2014.astype(int)[:, 0]
train_id_list_gen = train_id_gen.tolist()  # Necessary for parallel computing
train_id_imp = ID_2014_L.astype(int)[:, 0]
train_id_list_imp = train_id_imp.tolist()

# Get the ids for testing.
test_id_2015_unseen = numpy.setdiff1d(ID_2015, numpy.intersect1d(ID_2014, ID_2015)).astype(int)
test_id_2015_seen = numpy.intersect1d(ID_2014, ID_2015)
test_id_2015_imp = numpy.setdiff1d(ID_2015_L, numpy.intersect1d(ID_2014_L, ID_2015_L)).astype(int)
test_id_gen = test_id_2015_unseen.astype(int)
test_id_list_gen = test_id_gen.tolist()
test_id_imp = test_id_2015_imp.astype(int)
test_id_list_imp = test_id_imp.tolist()

id_set = [f for f in os.listdir(SRC_FOLDER) if os.path.isdir(os.path.join(SRC_FOLDER, f))]

# TODO: Calling multiple processors for creating pairs
if choice_phase == 'TEST':
    for ID in test_id_list_gen:
        Generate_Genuine(ID, SRC_FOLDER, DST_FOLDER_TEST,
                                  choice_phase, choice_feature, speech_aligned)
    for ID in test_id_list_imp:
        Generate_Impostor(ID, id_set, relations, SRC_FOLDER, DST_FOLDER_TEST, choice_phase, choice_feature,
                                   speech_aligned)
else:
    for ID in train_id_list_gen:
        Generate_Genuine(ID, SRC_FOLDER, DST_FOLDER_TRAIN,
                                  choice_phase, choice_feature, speech_aligned)
    for ID in train_id_list_imp:
        Generate_Impostor(ID, id_set, relations, SRC_FOLDER, DST_FOLDER_TRAIN, choice_phase, choice_feature,
                                   speech_aligned)
