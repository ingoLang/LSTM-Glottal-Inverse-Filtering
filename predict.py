import torch
import torchaudio
from model import BiLSTM
import os
import json

def loadModelData(model_path, hyperparameters, debug_mode):
    """
    Loads model data

    :param model_path:
    :param hyperparameters:
    :param debug_mode:
    :return:
    """

    # Initialize Model
    model = BiLSTM(
        hyperparameters,
        debug_mode

    )
    # Load Model
    model_data = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_data['model_state_dict'])
    return model


class Predictor:

    def __init__(self, hyperparameters, device):
        self.hyperparameters = hyperparameters

        self.input_size_lstm = hyperparameters["input_size_lstm"]  # Size
        self.output_sample_rate = hyperparameters["output_sample_rate"] # Sample rate of the DeepGIF Process (8kHz)
        self.criterion_name = hyperparameters["criterion"]
        self.device = device

    def preprocess(self, signal, old_sample_rate):
        """Preprocesses Files: Changing Samplerate
        """
        # Bring signal to float32
        signal = signal.float()

        # Change Sample-Rate from e.g.  44100Hz to output_sample_rate Hz
        new_sample_rate = self.output_sample_rate

        # Torch Audio to downsample signals: Algorithm: sinc-interpolation (no low-pass filtering needed)
        downsample = torchaudio.transforms.Resample(old_sample_rate,
                                                    new_sample_rate,
                                                    resampling_method='sinc_interpolation')
        down_sampled_signal = downsample(signal)
        return down_sampled_signal

    def loadSpeechSignal(self, speech_signal_file_address):
        speech_signal, sample_rate = torchaudio.load(speech_signal_file_address)
        return speech_signal, sample_rate

    def predict(self, model, speech_signal_address):
        # Load Signals
        speech_signal, speech_signal_sample_rate = self.loadSpeechSignal(speech_signal_address)

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

if __name__ == '__main__':
    # Set paths
    model_folder_address = 'Model'
    model_configuration_folder_address = 'ModelConfigurationFiles'
    model_name = 'Bi_LSTM_HiddenSize_30_LearnRate_0_01'
    dataset_address = 'Dataset'
    model_address = os.path.join(model_folder_address, str(model_name + '.pth'))
    model_configuration_file_address = os.path.join(model_folder_address,
                                                    model_configuration_folder_address,
                                                    str(model_name + '.json'))
    speech_signal_address = os.path.join(dataset_address, 'Berliner_001_Breathy', 'Berliner_001_Breathy_F0Offset0p241_PhoneRate0p734_NoNoise.wav')
    device = 'cpu'

    with open(model_configuration_file_address) as f:
        hyperparameters = json.load(f)
    print(hyperparameters)
    print('Model configuration with hyperparameters loaded!')

    model = loadModelData(model_address, hyperparameters, debug_mode=False)

    predictor = Predictor(hyperparameters, device)

    signals = predictor.predict(model, speech_signal_address) #signals = [speech_signal, predicted_glottal_flow_derivative_signal]

    print(signals)