import h5py
import dask.array as da
import numpy
import scipy.io as sio
import glob
import os
# import cv2
import sys
import itertools


def Generate_Impostor(ID, id_set, relations, src_folder, dst_folder, choice_phase, choice_feature,
                      speech_aligned):
    # Restrict the number of impostor pairs
    max_num_impostor_pairs = 30

    # Find the left and right ids in the relations.
    id_left = str(ID)
    id_right = str(relations[int(numpy.where(relations[:, 0] == ID)[0][0]), 1])
    if id_left in id_set and id_right in id_set:

        sound_left_all = glob.glob(os.path.join(src_folder, str(id_left), '*.npy'))
        sound_right_all = glob.glob(os.path.join(src_folder, str(id_right), '*.npy'))

        # Looping over all files in the id folders.
        counter = 0
        for sound_left_number, sound_left_name in enumerate(sound_left_all):
            for sound_right_number, sound_right_name in enumerate(sound_right_all):

                if speech_aligned == False:

                    # # Preventing repetition
                    if sound_left_number == sound_right_number and counter < max_num_impostor_pairs:
                        counter += 1
                        left = numpy.load(sound_left_name)
                        right = numpy.load(sound_right_name)

                        pair = numpy.concatenate((left[:, :, :, None], right[:, :, :, None]), axis=3)

                        # Save by the appropriate naming convention.
                        numpy.save(os.path.join(dst_folder,
                                                choice_phase + '_' + choice_feature + '_' + 'imp' + '_' + str(
                                                    ID) + '_' + str(
                                                    counter)), pair)
                        print(os.path.join(dst_folder,
                                           choice_phase + '_' + choice_feature + '_' + 'imp' + '_' + str(
                                               ID) + '_' + str(counter)))
                if speech_aligned == True:

                    if (os.path.basename(sound_right_name).split('.')[0]).split('_')[-1] == \
                            (os.path.basename(sound_left_name).split('.')[0]).split('_')[-1]:

                        counter += 1
                        if counter < num_impostor_pairs:
                            left = numpy.load(sound_left_name)
                            right = numpy.load(sound_right_name)

                            pair = numpy.concatenate((left[:, :, :, None], right[:, :, :, None]), axis=3)

                            # Save by the appropriate naming convention.
                            numpy.save(os.path.join(dst_folder,
                                                    choice_phase + '_' + choice_feature + '_' + 'imp' + '_' + str(
                                                        ID) + '_' + str(
                                                        counter)), pair)

                            print(os.path.join(dst_folder,
                                               choice_phase + '_' + choice_feature + '_' + 'imp' + '_' + str(
                                                   ID) + '_' + str(counter)))
