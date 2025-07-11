import torch
import csv
import json
import argparse
import time
import torch.utils.data as data
from torch import nn

from model import BiLSTM
from custom_dataset import GIFDataset
from predict import Predictor
from visualize_signals import Visualizer
from util import get_model_paths, get_signal_paths
from train import Trainer

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

def predict(model_name: str, signal_name: str, device: str):
    model_address, model_configuration_file_address, _ = get_model_paths(model_name)
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
    
def train(model_name, dataset_address, load_pretrained_model):
    model_address, model_configuration_file_address, model_folder_address = get_model_paths(model_name)
    
    # Load model configuration file
    with open(model_configuration_file_address) as f:
        hyperparameters = json.load(f)
    print(hyperparameters)
    print('Model configuration with hyperparameters loaded!')

    # Configure Device
    device = 'cpu'
    if torch.cuda.is_available():
        print('Torch.Cuda available!')
        device = 'cuda'
    else:
        print("Warning: No cuda device detected!")

    train_dataset = GIFDataset(dataset_address, hyperparameters, device, modus='Train')

    # Initialize dataloaders
    shuffle = True  # shuffle the dataset
    train_loader = data.DataLoader(train_dataset, hyperparameters["batch_size"], shuffle)
    print("Dataset loaded successfully!")

    # Initialize network and bring it to device
    model = BiLSTM(
        hyperparameters,
        debug_mode=False
    )

    if load_pretrained_model:
        model_data = torch.load(model_address)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer = torch.optim.SGD(model.parameters(), hyperparameters['learning_rate'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        loss_list = model_data['loss']
        print('Model loaded!')
    else:
        loss_list = []
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"])

    model.to(device)

    # Loss function for training
    # Squared error (squared L2 norm) between each element in the input xxx and target yyy
    if hyperparameters['criterion'] == 'MSE':
        criterion = nn.MSELoss(reduction="none")
        print('MSE-LOSS selected!')
    if hyperparameters['criterion'] == 'COSINE':
        criterion = nn.CosineEmbeddingLoss()
        print('COSINE-LOSS selected!')

    # Load the model trainer
    trainer = Trainer(hyperparameters,
                      model_name,
                      model_folder_address,
                      debug_mode=False,
                      device=device,
                      save_model_progress=False)

    # Start timer
    print("Start training!")
    t1 = time.time()
    
    
    # Train process
    train_signal_loss_list = trainer.train(model,
                                           criterion,
                                           optimizer,
                                           loss_list,
                                           train_loader,
                                           )

    # Stop the timer
    print("Finished in {0} seconds!".format(time.time() - t1))
    
    # Write the losses into a file
    header = 'losses'
    with open(model_folder_address + model_name + '_TrainResults.csv', 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows([[header]])
        for loss in train_signal_loss_list:
            writer.writerow([str(round(loss, 4))])
        writer.writerows([['Average Loss:']])
        writer.writerows([[str(sum(loss_list) / len(loss_list))]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--mode", required=False, default='predict')
    parser.add_argument("-m", "--model_name", required=False, default='Bi_LSTM_HiddenSize_30_LearnRate_0_01_TEST')
    parser.add_argument("-s", "--signal_name", required=False, default='Marburger_063_Modal_F0Offset2p857_PhoneRate0p846_WhiteNoise30db')
    parser.add_argument("-d", "--dataset_path", required=False, default='Dataset/')
    parser.add_argument("-l", "--load_pretrained_model", required=False, default=False)
    args = parser.parse_args()
    
    if args.mode == "predict":
        predict(args.model_name, args.signal_name, "cpu")
    elif args.mode == "train":
        train(args.model_name, 
                args.dataset_path, 
                args.load_pretrained_model,
            )