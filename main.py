import torch
from model import BiLSTM
import json
import argparse

from predict import Predictor
from visualize_signals import Visualizer
from util import get_model_paths, get_signal_paths

def load_model_data(model_path: str, hyperparameters: dict, device: str, debug_mode: bool) -> BiLSTM:
    # Initialize model
    model = BiLSTM(
        hyperparameters,
        debug_mode

    )
    # Load model data
    model_data = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_data['model_state_dict'])
    return model

def main(model_name: str, signal_name: str, device: str):
    model_address, model_configuration_file_address = get_model_paths(model_name)
    speech_signal_address, glottal_flow_signal_address, glottal_flow_derivative_signal_address = get_signal_paths(signal_name)
    
    with open(model_configuration_file_address) as f:
        hyperparameters = json.load(f)
    
    print(f"Model: {model_name}")
    print(f'Signal: {signal_name}')
    print('Model configuration with hyperparameters loaded:')
    print(hyperparameters)

    model = load_model_data(model_address, hyperparameters, device, debug_mode=False)

    predictor = Predictor(hyperparameters, device)
    
    [_, glottal_flow_derivative_signal, glottal_flow_signal, _] = predictor.load_synthetic_signals(speech_signal_address,
                                                                    glottal_flow_derivative_signal_address, 
                                                                    glottal_flow_signal_address,
                                                                    True)
    
    [speech_signal, predicted_glottal_flow_derivative_signal] = predictor.predict(model, speech_signal_address)

    sequence_length = 640 # Data points per plotted sequence
    stride = 120 # Stride in data points
    sequence_index = 28 # Which part/sequence of the signal do you want to plot
    save_figure = True 
    visualizer = Visualizer(sequence_length, stride, hyperparameters)
    visualizer.visualize_synthetic_signals([speech_signal, 
                                        glottal_flow_derivative_signal,
                                        glottal_flow_signal,
                                        predicted_glottal_flow_derivative_signal], 
                                        sequence_index, 
                                        save_figure,
                                        signal_name)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    parser.add_argument("-m", "--model_name", required=False, default='Bi_LSTM_HiddenSize_30_LearnRate_0_01')
    parser.add_argument("-s", "--signal_name", required=False, default='Marburger_063_Modal_F0Offset2p857_PhoneRate0p846_WhiteNoise30db')
    parser.add_argument("-d", "--device", required=False, default='cpu')
    args = parser.parse_args()
    
    main(args.model_name, args.signal_name, args.device)