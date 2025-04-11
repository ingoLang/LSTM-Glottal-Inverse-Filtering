import torch
import torchaudio
from model import BiLSTM
import os
import json

from predict import Predictor
from visualize_signals import Visualizer

def load_model_data(model_path, hyperparameters, debug_mode):
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



if __name__ == '__main__':
    # Set paths
    model_folder_address = 'Model'
    model_configuration_folder_address = 'ModelConfigurationFiles'
    model_name = 'Bi_LSTM_HiddenSize_30_LearnRate_0_01'
    
    signal_name = 'Berliner_001_Breathy_F0Offset0p241_PhoneRate0p734_NoNoise'
    
    model_address = os.path.join(model_folder_address, str(model_name + '.pth'))
    model_configuration_file_address = os.path.join(model_folder_address,
                                                    model_configuration_folder_address,
                                                    str(model_name + '.json'))
    
    dataset_address = 'Dataset'
    dir_address = "_".join(signal_name.split("_")[0:3])
    glottal_flow_name = "_".join(signal_name.split("_")[0:5]) + "_GlottalFlow.misc"
    glottal_flow_derivative_name = "_".join(signal_name.split("_")[0:5]) + "_GlottalFlowDerivative.misc"

    speech_signal_address = os.path.join(dataset_address, dir_address, signal_name + '.wav')
    glottal_flow_signal_address = os.path.join(dataset_address, dir_address, glottal_flow_name)
    glottal_flow_derivative_signal_address = os.path.join(dataset_address, dir_address, glottal_flow_derivative_name)
    
    device = 'cpu'

    with open(model_configuration_file_address) as f:
        hyperparameters = json.load(f)
    print(hyperparameters)
    print('Model configuration with hyperparameters loaded!')

    model = load_model_data(model_address, hyperparameters, debug_mode=False)

    predictor = Predictor(hyperparameters, device)

    speech_signal, glottal_flow_derivative_signal, glottal_flow_signal, _ = predictor.load_synthetic_signals(speech_signal_address,
                                                                                                    glottal_flow_derivative_signal_address, 
                                                                                                    glottal_flow_signal_address)
    [_, predicted_glottal_flow_derivative_signal] = predictor.predict(model, speech_signal_address)
    

    sequence_length = 640 #Abtastpunkte pro Sequenz
    stride = 120 #Schrittweite in sample rates
    visualizer = Visualizer(sequence_length, stride, hyperparameters)

    sequence_index = 14 # Which part/sequence of the signal do you want to plot
    save_figure = False 
    visualizer.visualize_synthetic_signals([speech_signal, 
                                        glottal_flow_signal,
                                        glottal_flow_derivative_signal,
                                        predicted_glottal_flow_derivative_signal], 
                                        sequence_index, 
                                        save_figure)