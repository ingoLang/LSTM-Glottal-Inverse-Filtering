import torch
from torch import nn


# Create Custom Bidirectional LSTM
class BiLSTM(nn.Module):

    def __init__(self, hyperparameters, debug_mode):
        super(BiLSTM, self).__init__()

        self.input_size_lstm = hyperparameters["input_size_lstm"]
        self.hidden_size_lstm = hyperparameters['hidden_size_lstm']
        self.number_layers_lstm = hyperparameters['number_layers_lstm']
        self.bidirectional_lstm = hyperparameters['bidirectional_lstm']

        if self.bidirectional_lstm:
            self.directions_lstm = 2
        else:
            self.directions_lstm = 1

        self.output_size_linear = hyperparameters['output_size_linear']
        self.debug_mode = debug_mode

        # LSTM-Layer
        self.lstm = nn.LSTM(
            input_size=self.input_size_lstm,
            hidden_size=self.hidden_size_lstm,
            num_layers=self.number_layers_lstm,
            batch_first=True,  # https://discuss.pytorch.org/t/could-someone-explain-batch-first-true-in-lstm/15402/4
            bidirectional=self.bidirectional_lstm

        )

        # Dense-Layer on top of LSTM
        self.linear = nn.Linear(self.hidden_size_lstm * self.directions_lstm, self.output_size_linear)


    def forward(self, input_data):
        """Defines the forward-Propagation logic
        Args:
            input_data ([Tensor]): [A 3-dimensional float tensor containing parameters]
            input_data of shape (batch, sequence_length, input_size):

        """
        # Feed Sequence into LSTM. Output is sequence
        output_lstm, (h_t, c_t) = self.lstm(input_data)

        # Reshaped Output LSTM Vector-Size: torch.Size([120, 60])
        reshaped_output_lstm = output_lstm.reshape(-1, self.hidden_size_lstm * self.directions_lstm)

        # Feed into Linear Layer/ Dense Layer.
        output_linear = self.linear(reshaped_output_lstm)

        # Squeeze Dimensions
        predictions = torch.squeeze(output_linear, 1)

        if self.debug_mode:
            print('Initial Hidden States Vector-Size:   ' + str(h_t.size()))
            print(h_t)
            print('Input Vector-Size:   ' + str(input_data.size()))
            print(input_data)
            print('Returned LSTM Hidden Vector-Size:   ' + str(h_t.size()))
            print(h_t)
            print('Output LSTM Vector-Size:   ' + str(output_lstm.size()))
            print(output_lstm)
            print('Reshaped Output LSTM Vector-Size:   ' + str(reshaped_output_lstm.size()))
            print('Output of Squeezed Fully-Connected-Layer-Size:   ' + str(predictions.size()))
            print(predictions)

        return predictions
