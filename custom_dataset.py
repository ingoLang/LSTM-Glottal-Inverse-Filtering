import torch
import torch.utils.data as data
import torchaudio
import pandas as pd
import os

class GIFDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, dataset_address, hyperparameters, device, modus):
        """Reads source and target sequences addresses from csv file."""
        self.dataset_address = dataset_address  # Location of dataset
        self.modus = modus  # Train or Test set
        self.path = self.dataset_address + self.modus + 'Files.csv'  # Creating Path to ID-List with all Files to train or test on.

        self.raw_id_list = open(self.path).readlines()  # Read ID List
        self.id_list = self.raw_id_list[1:len(self.raw_id_list)]  # Remove CSV-Header from List
        self.num_total = len(self.id_list)  # Total Number of SpeechSignal-GlottalFlowDeri- pairs
        self.output_sample_rate = hyperparameters["output_sample_rate"]

        self.device = device

    def __getitem__(self, index):
        """Returns one data pair (source and target).
        While shuffeling enabled, the dataloader randomizes the idx parameter
        of __getitem__, effectivelyy choosing a random file each time.
        Reads source and target sequences from wav files and misc files.
        """
        speech_signal_file_address, glottal_flow_derivative_file_address = self.getFilesAddress(index)
        channel = 0  # Only one channel!

        # Try to Load
        try:
            speech_signal, glottal_flow_derivative_signal, sample_rate = self.loadSignals(speech_signal_file_address,
                                                                                          glottal_flow_derivative_file_address)
            speech_signal = speech_signal[channel]
            glottal_flow_derivative_signal = glottal_flow_derivative_signal[channel]
            speech_signal, glottal_flow_derivative_signal = self.preprocess(speech_signal, glottal_flow_derivative_signal,
                                                                        sample_rate)

            if speech_signal.size() != glottal_flow_derivative_signal.size():
                print('Problems with signal length:' + speech_signal_file_address + ',' + glottal_flow_derivative_file_address)
                raise IndexError
                
        except:
            print('Error with loading files:' + speech_signal_file_address + ',' + glottal_flow_derivative_file_address)

        
        return speech_signal, glottal_flow_derivative_signal

    def __len__(self):
        """Returns the number of all signals in the test or train set.
        """
        return self.num_total

    def getFilesAddress(self, index):
        """Reads CSV-File and extracts addresses of speech-signal file and glottal flow derivative-Signal.
        """
        manipulation_name = str.rstrip(self.id_list[index].split(',')[1])
        subdir_name = "_".join(manipulation_name.split('_')[0:3])
        speech_signal_file_name = manipulation_name + '.wav'
        glottal_flow_derivative_file_name = manipulation_name + '_GlottalFlowDerivative.wav'
        speech_signal_file_address = os.path.join(self.dataset_address, subdir_name, speech_signal_file_name)
        glottal_flow_derivative_file_address = os.path.join(self.dataset_address, subdir_name, glottal_flow_derivative_file_name)
        return speech_signal_file_address, glottal_flow_derivative_file_address

    def loadSignals(self, speech_signal_file_address, glottal_flow_derivative_file_address):
        """Loads speech signal and glottal flow derivative signal
        """
        speech_signal, sample_rate = torchaudio.load(speech_signal_file_address)
        glottal_flow_derivative_signal, _ = torchaudio.load(glottal_flow_derivative_file_address)
        return speech_signal, glottal_flow_derivative_signal, sample_rate

    def preprocess(self, speech_signal, glottal_flow_derivative_signal, speech_signal_sample_rate):
        """Preprocesses Files: Changing Samplerate and Windowing
        """
        # Convert signal to float32 so training on gpu is faster
        # Bring whole signal to gpu
        speech_signal = speech_signal.float().to(self.device)
        glottal_flow_derivative_signal = glottal_flow_derivative_signal.float().to(self.device)

        # Change Sample-Rate from 44100Hz to 8000Hz
        new_sample_rate = self.output_sample_rate
        # Torch Audio to downsample signals: Algorithm: sinc-interpolation (no low-pass filtering needed)
        
        downsample = torchaudio.transforms.Resample(speech_signal_sample_rate,
                                                    new_sample_rate,
                                                    resampling_method='sinc_interp_hann')

        down_sampled_speech_signal = downsample(speech_signal)
        
        return down_sampled_speech_signal, glottal_flow_derivative_signal
