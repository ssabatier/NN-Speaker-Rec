import h5py
import dask.array as da
import numpy
import scipy.io as sio
import glob
import sys
import os
# import cv2
import sys
import itertools
import errno
'''
This file will generate cube of features. The procedure is as follows:

    1- The search will be done per id, i.e., for each id all the associate files
       with specific characteristics(belonging to which year or session) will be found.
       The associated file is the cube of features per id. So the format of the loaded
       numpy file is (Channels, features ,frames):
       Channels: static, first and second order derivatives.
       features: Number of extracted features(ex: 13 for MFCC)

    2- For each id the cubes of data will be generated each cube belongs to one a speech
       sound with some specific duration which has specific number of frames and will be saved
       with the naming convention associated with the frame id.
       ex: 1s sound has approximately 98 frames with frame length of 25ms and %60 overlap(i.e. 10ms stride).

    3- The final cube shape is (Channels, features ,frames) == (channel,height,width).
       features: is the number of extracted features.
       ex: The shape can be like (3,40,98) which 98 is the aforementioned number of frames.
'''

def Generate_Cube(id, src_folder, dst_folder, years, choice_phase, choice_feature, sessions, number_frames, overlap_stride):

    # TODO: Getting all the file names.
    feature_names_all = glob.glob(os.path.join(src_folder, choice_feature, str(id), str(
        id) + '*.npy')) # Get the all numpy files in the source folder and its subfolders.


    # Check if the image belongs to the desired year and session if not.
    for feature_name in feature_names_all:
        # if '*rainbow_VAD.npy' or '*0003_VAD.npy' in feature_names_all:
        if any(year in os.path.basename(feature_name).split('_')[1] for year in years):

            # We loop over each session to save the files in different sessions in similar but distinguishable names.
            # So the ids which apear in both session of a year will be saved as the last element of their name is an
            # identical number like "1" or "2" and however since the session name is saved too, so their name is different
            # using the different session name.
            for session in sessions:
                if session in os.path.basename(feature_name).split('_')[1]:
                    if 'rainbow_VAD' in feature_name or '0004_VAD' in feature_name:

                        # Format: (Channels, num_features ,frames)
                        # Frames is the number of frames
                        # num_features: Number of mel frequencies or extracted features.
                        # Channels: static, first and second order derivatives
                        feature_cube = numpy.load(feature_name)
                        print(feature_cube.shape)

                        # Number of all frames in the third dimension.
                        number_all_frames = feature_cube.shape[2]

                        id_folder = dst_folder + '/' + str(id)
                        try:
                            os.mkdir(id_folder)
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise exc
                            pass

                        # try:
                        # Number of stack is the number of stack of frames which form a block of data.
                        # For example: A 1-second block of data almost have 98 frames and the number of
                        # 1-second block of data in the stacked signal equals to whole number of frames
                        # divided by the number of frames per block.
                        number_stack = int(numpy.floor(number_all_frames / number_frames))

                        for i in range(number_stack):

                            # Get frame_feature_cube: The cube of features per sound file duration(ex: 1s sound file has 98 frames)
                            frame_feature_cube = feature_cube[ :, :,i * overlap_stride: i * overlap_stride + number_frames]
                            print("frame_feature_cube.shape", frame_feature_cube.shape)

                            # Save with the appropriate naming convention.
                            # if file_choice == 'rainbow':
                            numpy.save(os.path.join(dst_folder, id_folder,
                                                    choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                                                        id) + '_' + session + '_rainbow_' + str(i + 1)), frame_feature_cube)

                            print(os.path.join(dst_folder, id_folder,
                                                    choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                                                        id) + '_' + session + '_rainbow_' + str(i + 1)))
                            # elif file_choice == 'subject':
                            #     numpy.save(os.path.join(dst_folder, id_folder,
                            #                             choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                            #                                 id) + '_' + session + '_subject_' + str(i + 1)), frame_feature_cube)
                            #
                            #     print(os.path.join(dst_folder, id_folder,
                            #                             choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                            #                                 id) + '_' + session + '_subject_' + str(i + 1)))
                            # elif file_choice == 'both':
                            #     numpy.save(os.path.join(dst_folder, id_folder,
                            #                             choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                            #                                 id) + '_' + session + '_subject_' + str(i + 1)),
                            #                frame_feature_cube)
                            #
                            #     print(os.path.join(dst_folder, id_folder,
                            #                        choice_phase + '_' + choice_feature + '_' + 'cube' + '_' + str(
                            #                            id) + '_' + session + '_subject_' + str(i + 1)))
