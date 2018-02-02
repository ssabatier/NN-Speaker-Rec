import os
import glob
import sys
import numpy as np
from sklearn import metrics
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from easygui import *
import multiprocessing
import random
import dircache
import lmdb
import numpy as np
from PyQt4.QtGui import *
from PySide import QtGui, QtCore


class MyButtons(QtGui.QDialog):
    """"""

    def __init__(self, choices, title):
        # Initialized and super call.
        super(MyButtons, self).__init__()
        self.initUI(choices, title)
        self.choice = choices

    def initUI(self, choices, title):
        option1Button = QtGui.QPushButton(choices[0])
        option1Button.clicked.connect(self.onOption1)
        option2Button = QtGui.QPushButton(choices[1])
        option2Button.clicked.connect(self.onOption2)
        option3Button = QtGui.QPushButton(choices[2])
        option3Button.clicked.connect(self.onOption3)
        option4Button = QtGui.QPushButton(choices[3])
        option4Button.clicked.connect(self.onOption4)

        buttonBox = QtGui.QDialogButtonBox()
        buttonBox = QtGui.QDialogButtonBox(QtCore.Qt.Horizontal)
        buttonBox.addButton(option1Button, QtGui.QDialogButtonBox.ActionRole)
        buttonBox.addButton(option2Button, QtGui.QDialogButtonBox.ActionRole)
        buttonBox.addButton(option3Button, QtGui.QDialogButtonBox.ActionRole)
        buttonBox.addButton(option4Button, QtGui.QDialogButtonBox.ActionRole)
        #
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(buttonBox)

        self.setLayout(mainLayout)
        # define window		xLoc,yLoc,xDim,yDim
        self.setGeometry(250, 250, 100, 100)
        self.setWindowTitle(title)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    def onOption1(self):
        self.retStatus = 1
        self.close()
        self.choice = self.choice[0]

    def onOption2(self):
        self.retStatus = 2
        self.close()
        self.choice = self.choice[1]

    def onOption3(self):
        self.retStatus = 3
        self.close()
        self.choice = self.choice[2]

    def onOption4(self):
        self.retStatus = 4
        self.close()
        self.choice = self.choice[3]


"""
GUI for training or testing phase.
"""
app = QtGui.QApplication(sys.argv)
user_options = ['TRAIN', 'TEST', 'Cancel', 'Continue']
task_title = 'Are you intended to create testing or training pairs?!'
form = MyButtons(choices=user_options, title=task_title)
form.exec_()
choice_phase = form.choice

# If user canceled the operation.
if choice_phase == 'Cancel':
    sys.exit("Canceled by the user")

"""
GUI for getting the type of features.
"""
user_options = ['logfbank_energy', 'fbank_energy', 'MFCC', 'raw']
task_title = 'From which kind of features you want to create pairs?!'
form = MyButtons(choices=user_options, title=task_title)
form.exec_()
choice_feature = form.choice

# Source and destination paths(both with absolute path).
this_path = os.path.dirname(os.path.abspath(__file__))
Folder = '/media/sina/3F8C28A65EBC23F1/Research/Speech/Code/4-CreatePairs/PAIRS/' + choice_feature + '/' + choice_phase

"""
EXTRACTING THE DISTANCES
"""
files = glob.glob(os.path.join(Folder, '*.npy'))
number_of_files = len(files)

distance_original = []
distance_gen = []
distance_imp = []
label = []

for file in files:
    # Euclidean loss
    pair_1 = np.load(file)[:, :, :, 0]
    pair_2 = np.load(file)[:, :, :, 1]
    distance = np.sqrt(np.sum(np.square(pair_1 - pair_2)))
    distance_original.append(distance)

    if 'gen' in file:
        label.append(1)
        distance_gen.append(distance)
    elif 'imp' in file:
        label.append(0)
        distance_imp.append(distance)
    else:
        sys.exit('Something wrong with the file!')

# Return the numpy array
distance_original = np.asarray(distance_original)
dissimilarity = distance_original


"""
Plotting the histogram of original features and the ROC curve
"""
bins = np.linspace(0, int(np.amax(distance_original)), 50)
fig = plt.figure()
plt.hist(distance_gen, bins, alpha=0.5, facecolor='blue', normed=False, label='gen_dist_original')
plt.hist(distance_imp, bins, alpha=0.5, facecolor='red', normed=False, label='imp_dist_original')
plt.legend(loc='upper right')
plt.show()
fig.savefig(choice_phase + '_' + 'OriginalFeatures_Histogram.jpg')

# ROC
fpr, tpr, thresholds = metrics.roc_curve( label, -dissimilarity, pos_label = 1 )

# Calculating EER
intersect_x=fpr[np.abs(fpr-(1- tpr)).argmin(0)]
EER = intersect_x
print("EER = ",float(("{0:.%ie}" % 1).format(intersect_x)))

# AUC(area under the curve) calculation
AUC = np.trapz(tpr,fpr)
print("AUC = ",float(("{0:.%ie}" % 1).format(AUC)))

# Plot the ROC
fig = plt.figure()
ax = fig.gca()
lines = plt.plot(fpr, tpr, label='ROC Curve')
plt.setp(lines, linewidth=3, color='r')
ax.set_xticks(np.arange(0,1,0.1))
ax.set_yticks(np.arange(0,1,0.1))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Cutting the floating number
AUC = '%.2f' % AUC
EER = '%.2f' % EER

# Setting text to plot
plt.text(0.5, 0.5, 'AUC = ' + str(AUC), fontdict=None)
plt.text(0.5, 0.4, 'EER = ' + str(EER), fontdict=None)
plt.grid()
plt.show()
fig.savefig(choice_phase + '_' + 'OriginalFeatures_ROC.jpg')
