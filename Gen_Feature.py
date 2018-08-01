import numpy
import glob
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
# from Feature_Extract_Fn import ExtractFeature
# from SPEECH_SIGNAL_PROCESSING import processing
import speechpy

def Gen_Feature_Speech(id, src_folder, dst_folder, sessions, choice_stack, feature_type):
    """
    This function gets the ids and associated folders for generating features.
    The outputs are numpy arrays.

    Arg:
        id: The ids from which we extract features
        src_folder: The source folder for sound files
        dst_folder: The destination folder for saving the files
        sessions: First or second session 
        stack: Whether to stack the signals or not

    output:
        numpy array: (frames,feature_size)
        

This file will generate cubes of speech features. The procedure is as follows:

        1- The search will be done per id, i.e., for each id all of the associated files
           with specific characteristics(belonging to which year or session) will be found.
           The search is done to find the sound files that have been pre-processed (VAD and re-sampled)
           by MATLAB and have 'preprocessed' string in their file names.

        2- Each sound file will be read and stacking of the frames will be done using "processing.Stack_Frames".

        3- For each stacked sequence of frames, the feature vectors(static, first, second order
           derivatives) will be extracted with shape: (frames,feature_size). The feature_size will
           be called Mels because it is associated with the number of chosen mel-frequencies.

        4- Using the static feature and derivatives, the feature cube will be generated with
           (frames, Mels ,Channels) dimensions but will be re-ordered to
           (Channels, Mels ,frames) for convenience.

        5- The final cube shape is (Channels, Mels ,frames) == (channel,height,width).
           * Mels is the number of filterbanks/specific frequencies.
           * Channels contain the following:
              channel[0]: static features
              channel[1]: first order derivatives
              channel[2]: second order derivatives
    """
    normalize_status = True
    for session in sessions:
        
        # Generating Genuine pairs
        # Nuemann and Neumann have different spelling in 2014 and 2015 folders
        
        if '2014' in session:
            file_names = glob.glob(os.path.join(src_folder, str(id), session,'Nuemann Mic','Audio_VAD','*.wav'))
                                                # '*VAD.wav'))  # Get the all image names in the source folder and its subfolders.
            # print(file_names)
        elif '2015' in session:
            file_names = glob.glob(os.path.join(src_folder, str(id), session,'Neumann Mic','Audio_VAD','*.wav'))
                                                # ,'*VAD.wav'))
        else:
            file_names = glob.glob(os.path.join(src_folder, str(id), session, 'Neumann Mic', 'Audio_VAD', '*.wav'))

        # Looping through all of the files
        for file_name in file_names:

            try:
                # Read the signal
                fs, signal = wav.read(file_name)
                signal = signal.astype(float)
                
                # Normalize signal
                if normalize_status:
                    signal = signal / 32767
                    # print(signal)
                    
                # Get the id name
                ID_name = os.path.basename(file_name).split('_')[0]

                # Create the corresponding folder
                ID_Folder = dst_folder + '/' + ID_name
                if not os.path.exists(ID_Folder):
                    os.makedirs(ID_Folder)

                # Stack frames
                # if choice_stack == 'YES':
                #     frame_length = 0.020
                #     overlap_factor = 0.0
                #     signal = processing.Stack_Frames(signal, fs, frame_length, overlap_factor,
                #                                      Filter=lambda x: np.ones((x,)),
                #                                      zero_padding=True)
                
                if feature_type == 'MFEC':
                    #MFEC Features---------------
                    static_feature = speechpy.feature.lmfe(signal, fs, frame_length=0.025, frame_stride=0.010, num_filters=40,
                                  fft_length=512, low_frequency=0, high_frequency=None)
                    # print(static_feature)

                elif feature_type == 'MFCC':
                    # #MFCC Features----------------------
                    static_feature = speechpy.feature.mfcc(signal, fs, frame_length=0.025, frame_stride=0.010, num_cepstral=13, num_filters=40,
                                            fft_length=512, low_frequency=0, high_frequency=None)
                else:
                    static_feature = signal

                # Feature Extraction
                # static_feature = ExtractFeature(signal, fs, feature_type)
                feature = speechpy.feature.extract_derivative_feature(static_feature)
                # first_derivative_feature = processing.Derivative_Feature_Fn(static_feature, DeltaWindows=2)
                # second_derivative_feature = processing.Derivative_Feature_Fn(first_derivative_feature, DeltaWindows=2)

                # # Creating the feature cube for each file
                # feature = np.concatenate(
                #     (static_feature[:, :, None], first_derivative_feature[:, :, None],
                #      second_derivative_feature[:, :, None]),
                #     axis=2)
                # print(feature.shape)

                # Reorder the dimension in a way to maintain locality
                # the (frames, Mels ,Channels) dimensions which will be re-ordered to (Channels, Mels ,frames).
                feature_cube = np.transpose(feature, (2, 1, 0))
                print(feature_cube.shape)

                # # # Uncomment if using numpy.reshape is desired # # #
                # # For reading the first index change faster(order: 'F'). It means the frames dimension
                # # will be read first.
                # feature_flat = np.ravel(feature, 'F')
                #
                # # For writing the last index change faster(order: 'C'). It means the frames dimension
                # # will be write first.
                # feature_cube = np.reshape(feature_flat, (feature.shape[2], feature.shape[1], feature.shape[0]), 'C')
                
  # Save the features of all files associated with a specific id, in the specific forlder.

                # # Uncomment if saving original features is necessary
                # np.save(os.path.join(ID_Folder, os.path.basename(file_name).split('.')[0]) + '_' + 'raw',
                #            signal)

                # # Uncomment if saving static, first and second order features in defferent files is necessary.
                # np.save(os.path.join(ID_Folder, os.path.basename(file_name).split('.')[0])+'_'+'Static',
                #            static_feature)
                # np.save(os.path.join(ID_Folder, os.path.basename(file_name).split('.')[0])+'_'+'Delta',
                #            first_derivative_feature)
                # np.save(os.path.join(ID_Folder, os.path.basename(file_name).split('.')[0])+'_'+'DeltaSquare',
                #            second_derivative_feature)
                
                output_file_name = os.path.join(ID_Folder, os.path.basename(file_name).split('.')[0])
                np.save(output_file_name,
                        feature_cube)
                print(('Print file {}').format(output_file_name,
                        feature_cube))
                # print(os.path.basename(file_name).split('.')[0], feature_cube.shape)
            except:
                print("Perhaps file is corrupted or damaged", output_file_name)
