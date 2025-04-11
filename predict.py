import torch
import torchaudio
import pandas as pd
import numpy as np


class Predictor:

    def __init__(self, hyperparameters, device):
        self.hyperparameters = hyperparameters

        self.input_size_lstm = hyperparameters["input_size_lstm"]  # Size
        self.output_sample_rate = hyperparameters["output_sample_rate"]
        self.criterion_name = hyperparameters["criterion"]
        self.device = device

    def load_synthetic_signals(self, speech_signal_address, glottal_flow_derivative_signal_address,
                             glottal_flow_signal_address):
        """Load Files the same way, like in GIFDataset-Class.
        In Addition, load glottal flow signal
        """
        try:
            speech_signal, speech_signal_sample_rate = torchaudio.load(speech_signal_address)
            channel = 0
            speech_signal = speech_signal[channel]

            glottal_flow_derivative_data = pd.read_csv(glottal_flow_derivative_signal_address, sep=' ')
            glottal_flow_derivative = glottal_flow_derivative_data['glottal_flow_derivative[cm^3/s^2]']
            glottal_flow_derivative_signal = torch.tensor(glottal_flow_derivative.values)

            glottal_flow_data = pd.read_csv(glottal_flow_signal_address, sep=' ')
            glottal_flow = glottal_flow_data['glottal_flow[cm^3/s]']
            glottal_flow_signal = torch.tensor(glottal_flow.values)


        except:
            print('Error with loading files:' + speech_signal_address + ',' + glottal_flow_derivative_signal_address)
            return

        if speech_signal.size() != glottal_flow_derivative_signal.size():
            print('Problems with files:' + speech_signal_address + ',' + glottal_flow_derivative_signal_address)
            # In the synthetic dataset all files have the same sample-rate, but better check
        return speech_signal, glottal_flow_derivative_signal, glottal_flow_signal, speech_signal_sample_rate

    def load_EGGsignal(self, egg_signal_address):
        """Load and preprocess Files the same way, like in GIFDataset-Class.
        """
        try:
            egg_signal, egg_signal_sample_rate = torchaudio.load(egg_signal_address)  # Load EGG Signals
            return egg_signal[0], egg_signal_sample_rate

        except:
            print('Error with loading files:' + egg_signal_address)
            return

    def load_speech_signal(self, speech_signal_address):
        """Load and preprocess Files the same way, like in GIFDataset-Class.
        """
        try:
            speech_signal, speech_signal_sample_rate = torchaudio.load(speech_signal_address)
            channel = 0
            speech_signal = speech_signal[channel]

            return speech_signal, speech_signal_sample_rate

        except:
            print('Error with loading files:' + speech_signal_address)
            return

    def preprocess(self, signal, sample_rate):
        """Preprocesses Files: Changing Samplerate and Windowing
        """
        # Bring signal to float32
        signal = signal.float()
        # Change Sample-Rate from 44100Hz to output_sample_rate Hz
        new_sample_rate = self.output_sample_rate
        # Torch Audio to downsample signals: Algorithm: sinc-interpolation (no low-pass filtering needed)
        downsample = torchaudio.transforms.Resample(sample_rate,
                                                    new_sample_rate,
                                                    resampling_method='sinc_interpolation')
        down_sampled_signal = downsample(signal)

        # print(down_sampled_speech_signal.dtypes)

        return down_sampled_signal

    def calculate_derivative(self, signal):
        # For derivative use numpy (For Pytorch recommended!)
        derivative_signal = torch.from_numpy(np.gradient(signal.cpu().numpy()))
        return derivative_signal


    def predict_synthetic_signals(self, model, criterion, speech_signal_address, glottal_flow_derivative_signal_address,
                                glottal_flow_signal_address):

        # Load Signals
        speech_signal, glottal_flow_derivative_signal, glottal_flow_signal, sample_rate = self.load_synthetic_signals(
            speech_signal_address, glottal_flow_derivative_signal_address, glottal_flow_signal_address)

        # Preprocess Signals
        # Returns a downsampled signal. Specify "output_sample_rate" in hyperparameters-data
        speech_signal = self.preprocess(speech_signal, sample_rate)
        glottal_flow_derivative_signal = self.preprocess(glottal_flow_derivative_signal, sample_rate)
        glottal_flow_signal = self.preprocess(glottal_flow_signal, sample_rate)

        # Bring to Device
        speech_signal = speech_signal.to(self.device)
        glottal_flow_derivative_signal = glottal_flow_derivative_signal.to(self.device)

        # Preallocate
        predicted_glottal_flow_derivative_signal = torch.tensor([]).to(self.device)
        signal_length = speech_signal.size(0)

        with torch.no_grad():
            # Reshape Tensor
            input_data = speech_signal.view(-1, signal_length, self.input_size_lstm)

            # Feed Data into Model
            predicted_glottal_flow_derivative_signal = model(input_data)

            if self.criterion_name == "MSE":
                loss = criterion(predicted_glottal_flow_derivative_signal, glottal_flow_derivative_signal)
            if self.criterion_name == "COSINE":
                y = torch.ones(1).to(self.device)  # y is control parameter for the cosine distance error
                loss = criterion(predicted_glottal_flow_derivative_signal.view(-1, 1),
                                 glottal_flow_derivative_signal.view(-1, 1),
                                 y.view(1).to(self.device))
            signal_loss = loss.mean().item()

        signals = [speech_signal, glottal_flow_derivative_signal, glottal_flow_signal,
                   predicted_glottal_flow_derivative_signal]

        return signals, signal_loss

    def predict(self, model, speech_signal_address):
        # Load Signals
        speech_signal, speech_signal_sample_rate = self.load_speech_signal(speech_signal_address)

        # Returns a downsampled signal. Specify "output_sample_rate" in hyperparameters-data
        speech_signal = self.preprocess(speech_signal, speech_signal_sample_rate)

        # Bring to Device
        speech_signal = speech_signal.to(self.device)

        # Preallocate
        predicted_glottal_flow_derivative_signal = torch.tensor([]).to(self.device)
        signal_length = speech_signal.size(0)

        with torch.no_grad():
            # Reshape Tensor
            input_data = speech_signal.view(-1, signal_length, self.input_size_lstm)

            # Feed Data into Model
            predicted_glottal_flow_derivative_signal = model(input_data)

        signals = [speech_signal, predicted_glottal_flow_derivative_signal]

        return signals

    def predict_and_get_EGGsignal(self, model, speech_signal_address, egg_signal_address):
        # Load Signals
        speech_signal, speech_signal_sample_rate = self.load_speech_signal(speech_signal_address)
        egg_signal, egg_signal_sample_rate = self.load_EGGsignal(egg_signal_address)

        # Returns a downsampled signal. Specify "output_sample_rate" in hyperparameters-data
        speech_signal = self.preprocess(speech_signal, speech_signal_sample_rate)
        egg_signal = self.preprocess(egg_signal, egg_signal_sample_rate)

        egg_derivative_signal = self.calculate_derivative(egg_signal)

        # Bring to Device
        speech_signal = speech_signal.to(self.device)
        egg_signal = egg_signal.to(self.device)

        # Preallocate
        predicted_glottal_flow_derivative_signal = torch.tensor([]).to(self.device)
        signal_length = speech_signal.size(0)

        with torch.no_grad():
            # Reshape Tensor
            input_data = speech_signal.view(-1, signal_length, self.input_size_lstm)

            # Feed Data into Model
            predicted_glottal_flow_derivative_signal = model(input_data)

        signals = [speech_signal, predicted_glottal_flow_derivative_signal,  egg_signal,
                   egg_derivative_signal]

        return signals
