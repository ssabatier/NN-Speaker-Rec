## siamese LSTM using keras

# # Investigate
# np.random.seed(1337)  # for reproducibility

import numpy as np
# import gzip, cPickle
import random
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.callbacks import CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.preprocessing import MinMaxScaler
import h5py
import sys



def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)


def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def l1_distance(vects):
    x, y = vects
    return K.abs(K.sum(x - y))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# # # # # Commented by me ############
# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 2
#     loss = y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
#     print loss.shape
#     return K.mean(loss)
#
# def robust_contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 4.0
#     return K.mean(y_true * K.square(K.minimum(margin, y_pred)) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# def create_pairs(x, digit_indices, n_cat=10):
#     '''Positive and negative pair creation.
#     Alternates between positive and negative pairs.
#     '''
#     pairs = []
#     labels = []
#     n = min([len(digit_indices[d]) for d in range(n_cat)]) - 1
#     if n==0:
#     	print("Eliminate some classes")
#     	cats = [len(digit_indices[d]) for d in range(n_cat)]
#     	elim_list = [i for i,j in enumerate(cats) if j == 1]
#     	cats = [i for j, i in enumerate(cats) if j not in elim_list]
#     	n = min(cats) - 1
#     	n_cat = len(cats)
#     	digit_indices = [i for j, i in enumerate(digit_indices) if j not in elim_list]
#     for d in range(n_cat):
#         for i in range(n):
#             z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
#             pairs += [[x[z1], x[z2]]]
#             inc = random.randrange(1, n_cat)
#             dn = (d + inc) % n_cat
#             z1, z2 = digit_indices[d][i], digit_indices[dn][i]
#             pairs += [[x[z1], x[z2]]]
#             labels += [1, 0]
#     return np.array(pairs), np.array(labels)


# # # # # # # Commented by me ############
# def compute_accuracy(predictions, labels):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return labels[predictions.ravel() < 0.5].mean()
#
# def compute_accu(labels, y_pred):
#     ''''Compute classification accuracy with a fixed threshold on distances.'''
#     #return K.mean(K.lesser(y_pred,K.ones(shape=(y_pred.get_shape()))))
#     return K.mean([K.get_value(y_pred) < 0.5])


def create_siamese_LSTM(input_dim):
    """
    Creating the architecture for siamese LSTM.

    :param input_dim: Looks like it is not defined in this function!!!
    :return: A (?,128) tensor as the output.

    PROCEDURE:

            1 - Create Keras Sequential model
            2 - Add layers one by one.
    """
    Sequential_Model = Sequential()

    """
    Add one dimensional convolutional models.

    Notes:
         * Activations are in the conv layers
         * The last conv layer has activations too.

    """
    # Convolution-1
    Sequential_Model.add(Convolution1D(filters=16, kernel_size=64, padding='valid', activation='relu',
                                       strides=1, input_shape=(6400, 1)))
    Sequential_Model.add(MaxPooling1D(pool_size=4))

    # Convolution-2
    Sequential_Model.add(Convolution1D(filters=32, kernel_size=32, padding='valid', activation='relu',
                                       strides=1))
    Sequential_Model.add(MaxPooling1D(pool_size=4))

    # Convolution-3
    Sequential_Model.add(Convolution1D(filters=64, kernel_size=16, padding='valid', activation='relu',
                                       strides=1))
    Sequential_Model.add(MaxPooling1D(pool_size=2))

    # Convolution-4
    # No pooling after last convolutional layer.
    # There is still activation for this layer.
    Sequential_Model.add(Convolution1D(filters=128, kernel_size=8, padding='valid', activation='relu',
                                       strides=1))

    """
    Add LSTM layer.
    """
    # The first LSTM returns its full output sequece.
    Sequential_Model.add(LSTM(128, return_sequences=True))

    # Dropout parameter.
    Sequential_Model.add(Dropout(0.5))

    # This LSTM returns only the last step in its output. So it drops the temporal dimension.
    Sequential_Model.add(LSTM(128))

    return Sequential_Model


# digit_indices_tr=[]
# # speakers verification
# for speak in subj_tr:
# 	digit_indices_tr.append([i for i, s in enumerate(ids) if speak in s ])
# tr_pairs, tr_y = create_pairs(prova, digit_indices_tr, len(subj_tr))

# digit_indices_te=[]
# # speakers verification
# for speak in subj_te:
# 	digit_indices_te.append([i for i, s in enumerate(ids) if speak in s ])
# te_pairs, te_y = create_pairs(prova, digit_indices_te, len(subj_te))

# dataset=[tr_pairs, tr_y, te_pairs, te_y]
# f=gzip.open('Speech_wav_stacked100_train_test.pkl.gz','wb')
# cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
# f.close()


"""
Part1: Input the data.
"""
hf_train = h5py.File('TRAIN.hdf5', 'r')
hf_test = h5py.File('TEST.hdf5', 'r')
print('List of arrays: \n', hf_train.keys())
tr_pairs = hf_train.get('pairs')
tr_y = hf_train.get('labels')
te_pairs = hf_test.get('pairs')
te_y = hf_test.get('labels')

# # The test subjects ids.
# subj_tr = hf.get('subj_tr')
# subj_te = hf.get('subj_te')

# Training data shape of format of (Number of pairs(samples), 2, X1, X2)
# X1: Probably temporal sequence, i.e., 100 successive frames.
# X2: The features per windows: 6400 features means 400ms if the audio is samples at 16000Hz.
print ("Training shape:", tr_y.shape)

# Turn into numpy arrays for processing.
tr_pairs = np.array(tr_pairs)
te_pairs = np.array(te_pairs)
tr_y = np.array(tr_y)
te_y = np.array(te_y)

# Required parameters.
input_dim = 6400
nb_epoch = 5

# network definition
base_network = create_siamese_LSTM(input_dim)

# Place holders.
input_a = Input(shape=(6400, 1))
input_b = Input(shape=(6400, 1))

# Output from two identical networks.
processed_a = base_network(input_a)
processed_b = base_network(input_b)


# Of shape processed_a.get_shape() and processed_a.get_shape()
abs_diff = Lambda(get_abs_diff, output_shape=abs_diff_output_shape)([processed_a, processed_b])

# Of shape (?, 1)
flattened_weighted_distance = Dense(1, activation='sigmoid')(abs_diff)

# Create the keras model.
model = Model(inputs=[input_a, input_b], outputs=flattened_weighted_distance)

# Otimization object.
rms = RMSprop()
sgd = SGD(lr=0.01)
adam = Adam()

# Model compiling.
#model = load_model('my_model_48_epoch.h5')
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
csv_logger = CSVLogger('training_rms_no_last_dense.log')

TensorBoard(log_dir='./logs/', histogram_freq=5, write_graph=True, write_images=True)
# For as (number of samples, number of features, 1). ex: (800,6400,1)

train_pair1 = tr_pairs[:, :, :, 0]
train_pair2 = tr_pairs[:, :, :, 3]
test_pair1 = te_pairs[:, :, :, 0]
test_pair2 = te_pairs[:, :, :, 3]

train_result = model.fit([train_pair1, train_pair2], tr_y,
                         validation_data=([test_pair1, test_pair2], te_y),
                         batch_size=500, epochs=nb_epoch, callbacks=[csv_logger])

model.save('my_model_'+ str(nb_epoch) +'_epoch.h5')

# compute final accuracy on training and test sets
tr_pred = model.predict(
    [train_pair1, train_pair2])
tr_acc = accuracy(tr_y, tr_pred.round())

te_pred = model.predict(
    [test_pair1, test_pair2])
te_acc = accuracy(te_y, te_pred.round())

print('* Accuracy on the training set: {:.2%}'.format(tr_acc))
print('* Accuracy on the test set: {:.2%}'.format(te_acc))
