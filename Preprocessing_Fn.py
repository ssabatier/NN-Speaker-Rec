import sys
import os
import numpy as np
from sklearn import preprocessing

'''
A model shall be applied on unseen data which is in general not available at the time the model is built.
The validation process (including data splitting) simulates this. So in order to get a good estimate of the
model quality (and generalization power) one needs to restrict the calculation of the normalization parameters
(mean and variance) to the training set.

The following functions are used for preprocessing the training data:

     # If using numpy is desired:
         Zero_Mean_Numpy: Make the data centred on zero. For each channel we will have a mean image(not a single value!)
         Normalization_Numpy: Normalization of the zero-mean data.

     # If using Scikit-learn is desired.
        Preprocessing_Scikit_Train: Mean subtraction + Scaling(normalization)
'''
def Zero_Mean_Numpy(X):
    '''
    Mean subtraction is the most common phase of preprocessing. It involves subtracting
    the mean of every individual feature in the data with the geometric interpretation
    of "centering the data around the origin in each dimension". In numpy, this operation
    would be implemented as: X -= np.mean(X, axis = 0) if the samples are within first
    dimension(ex: data size:(N,D) in which N is the samples and D is the number of dimensions).
    Considering speech specifically, it can be reasonable to subtract separately across the
    three channels(static,first and second order derivatives).

    :param X: The data cube.
    :return: X: The zero mean data cube
             mean_array: The mean array which is the mean for each channel
    '''
    X_Preprocessed = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    mean_array = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
    # Mean subtraction/channel.
    for i in range(3):
        mean_array[i, :, :] = np.mean(X[:, i, :, :], axis=0)
        X_Preprocessed[:, i, :, :] = X[:, i, :, :] - mean_array[i, :, :]

    return X_Preprocessed, mean_array

def Normalization_Numpy(X):
    '''
    "Normalization" refers to normalizing the data dimensions to be nearly in the same scale.
    The most common way is to divide each dimension by its standard deviation, after it has been
    zero-centered: (X /= np.std(X, axis = 0)) again if the samples are within first  dimension.
    It makes sense to apply "Normalization" if you have there is a reason to believe that different
    input features have different scales plus they are equally important for the learning procedure.
    For image this is not the case and values in three channels looks like to have the same scale
    but for speech this is not the case and different channels[static, first and second order derivatives]
    have different scaling so the "Normalization" looks necessary.

    :param X: The data cube.
    :return:
    X: The data cub which is zero mean!
    X: The normalized data cub!

    '''
    X_Preprocessed = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    std_array = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
    for i in range(3):
        std_array[i, :, :] = np.std(X[:, i, :, :], axis=0)
        X_Preprocessed[:, i, :, :] = X[:, i, :, :] / std_array[i, :, :]
    return X_Preprocessed, std_array


def Preprocessing_Scikit_Train(X, transform_status = False):
    """
    Preprocessing data using scikit-learn package.
    :param X: Input cube of size(N,Channels,Height,Width)
    :return: X: The zero mean and scaled data cube with the same size.
             mean_image: The mean of feature which is an image.
             std_image: The std of feature which is an image.

    """
    mean_image = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
    std_image = np.zeros((X.shape[1], X.shape[2], X.shape[3]))


    for i in range(3):

        X_Preprocessed = False

        # Getting the desired channel.
        X_Channel = X[:, i, :, :]

        # Transform each (width , height) image into a feature vector of length (width*height)
        X_Channel_reshaped = np.reshape(X_Channel, (X_Channel.shape[0], -1))

        # Using scaler API and returning mean and standard daviation array.
        scaler = preprocessing.StandardScaler().fit(X_Channel_reshaped)

        # Reformat the mean array to image of means.
        mean_array = scaler.mean_
        mean_image[i, :, :] = np.reshape(mean_array, (mean_image.shape[1], mean_image.shape[2]))

        # Reformat the std array to image of stds.
        std_array = scaler.scale_
        std_image[i, :, :] = np.reshape(std_array, (std_image.shape[1], std_image.shape[2]))

        # The transformation is done only if it is necessary because it is time-consuming.
        if transform_status:
            X_Preprocessed = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

            # Transform the data into zero-mean + std = 1 space.
            X_Channel_reshaped = scaler.transform(X_Channel_reshaped)

            # Reformating to original form
            X_Channel = np.reshape(X_Channel_reshaped, (X_Channel_reshaped.shape[0], X_Channel.shape[1], X_Channel.shape[2]))

            # Changing the desired channel with the new processed data.
            X_Preprocessed[:, i, :, :] = X_Channel

    return mean_image, std_image, X_Preprocessed

def TestData_Transfer_Fn(X, mean_image, std_image):
    '''
    By using this function, the test data would be transferred to the domain of training data
    using the mean and std of the features of training data.

    :param  X: The input data cube(which is the test data).
    :param  mean_array: The mean array calculated by training set
    :return:
           X_Postprocess: The test data which has been transferred to train domain.
    '''
    X_Postprocess = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    # Post processing per channel.
    for i in range(3):
        X_Postprocess[:, i, :, :] = X[:, i, :, :] - mean_image[i, :, :]
        X_Postprocess[:, i, :, :] = X_Postprocess[:, i, :, :] / std_image[i, :, :]

    return X_Postprocess

def File_Transfer_Fn(X, mean_image, std_image):
    '''
    By using this function, the test file would be transferred to the domain of training data
    using the mean and std of the features of training data.

    :param  X: The input data cube(which is a single file).
    :param  mean_array: The mean array calculated by training set
    :return:
           X_Postprocess: The test data which has been transferred to train domain.
    '''
    X_Postprocess = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    # Post processing per channel.
    for i in range(3):
        X_Postprocess[i, :, :] = X[i, :, :] - mean_image[i, :, :]
        X_Postprocess[i, :, :] = X_Postprocess[i, :, :] / std_image[i, :, :]

    return X_Postprocess
