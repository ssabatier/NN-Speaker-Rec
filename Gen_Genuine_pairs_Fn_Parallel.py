import h5py
import dask.array as da
import numpy
import scipy.io as sio
import glob
import os
# import cv2
import sys
import itertools

def Generate_Genuine(id, src_folder, dst_folder, choice_phase, choice_feature,
                     speech_aligned):
    """
    This function creates the genuine pairs based on the files created in "src_folder". Each file in "src_folder"
    is a block of frames which forms a cube. For example if we created a block of 1sec duration, then each second
    has 98 frames and 40 features, each block in "src_folder" has (3,40,98) dimensions. The naming
    convention for each file is preprocessed_choice_phase_choice_feature_cube_id_counter.
    
    args:
    id: each person id.
    src_folder: The folder which the pre-processed file will be read
    dst_folder: The destination folder which the files will be saved
    choice_phase: TRAIN or TEST
    choice_feature: MFCC, log-filterbank_energy (MFEC), etc. 
    speech_aligned: For creating genuine pairs, if each pair is built on the same speech words(the speaker utter
                    the in different sessions for example) then we are creating speech_aligned pairs otherwise the 
                    are speech_not_aligned and the network is learning speaker information.

    outputs:
    Saved pairs. Blocks are concatenated in the first axis. For example if each block is of size
    (channels,features,frames), then the pairs are of size (channels,features,frames,2)
    """
    # Restriction of maximum of genuine pairs per id
    num_genuine_pairs = 30

    # Generate Genuine pairs
    try:
        feature_names_all = glob.glob(os.path.join(src_folder, str(id),
                                                   '*.npy'))  # Get all numpy files in the source folder and its subfolders
        first_file = numpy.load(feature_names_all[0])

        # number_blocks is the number of files in folder which for example are the number of 1s slices of the data cube
        # ex: How many (3,40,98) blocks are there?
        number_blocks = len(feature_names_all)

        # Initialize the vector
        feature = numpy.zeros((number_blocks, first_file.shape[0], first_file.shape[1], first_file.shape[2]))

        if speech_aligned == False:
            counter = 0
            # Create genuine pair based on the same file
            # Check if the file belongs to the desired year or not
            for feature_name in feature_names_all:
                feature[counter, :, :, :] = numpy.load(feature_name)
                counter += 1

            # Generate random vector
            random_vector_1 = numpy.random.randint(number_blocks, size=(1, number_blocks))
            random_vector_2 = numpy.random.randint(number_blocks, size=(1, number_blocks))

            for i in range(num_genuine_pairs):
                # WARNING: In order to create effective genuine pairs, the block of data must be non-overlapped!
                left = feature[random_vector_1[0, i], :, :, :]
                right = feature[random_vector_2[0, i], :, :, :]
                pair = numpy.concatenate((left[:, :, :, None], right[:, :, :, None]), axis=3)

                # Save with the appropriate naming convention
                numpy.save(os.path.join(dst_folder,
                                        choice_phase + '_' + choice_feature + '_' + 'gen' + '_' + str(
                                            id) + '_' + str(
                                            str(i + 1))), pair)
                print(os.path.join(dst_folder,
                                   choice_phase + '_' + choice_feature + '_' + 'gen' + '_' + str(
                                       id) + '_' + str(
                                       str(i + 1))))
        else:
            counter = 0
            for file_name_L in feature_names_all:
                left = numpy.load(file_name_L)
                for file_name_R in feature_names_all:
                    if file_name_L != file_name_R:

                        # (os.path.basename(file_name_R).split('.')[0]).split('_')[-1] get the file name without extension(.npy)
                        # and then split by '_' and return the last number.
                        if (os.path.basename(file_name_R).split('.')[0]).split('_')[-1] == \
                                (os.path.basename(file_name_L).split('.')[0]).split('_')[-1]:
                            # Increase the counter
                            counter += 1

                            # Load the right part of the pair
                            right = numpy.load(file_name_R)

                            # Concatenate alongside the channel axis
                            pair = numpy.concatenate((left[:, :, :, None], right[:, :, :, None]), axis=3)

                            # Save by the appropriate naming convention
                            numpy.save(os.path.join(dst_folder,
                                                    choice_phase + '_' + choice_feature + '_' + 'gen' + '_' + str(
                                                        id) + '_' + str(
                                                        counter)), pair)

                            print(os.path.join(dst_folder,
                                               choice_phase + '_' + choice_feature + '_' + 'gen' + '_' + str(
                                                   id) + '_' + str(
                                                   counter)))

    except:
        print("The ID" + " " + str(id) + " " + "has been skipped.")
