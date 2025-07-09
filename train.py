import torch


class Trainer:

    def __init__(self, hyperparameters, model_name, model_folder_address, debug_mode, device, save_model_progress):
        self.hyperparameters = hyperparameters

        self.model_folder_address = model_folder_address  # Directory for temporary models
        self.model_name = model_name
        self.model_address = model_folder_address + model_name + '.pth'  # Path to final model

        self.number_epochs = hyperparameters["number_epochs"]
        self.sequence_length = hyperparameters["sequence_length"]  # Number of Samples in one Sequence
        self.window_stride = hyperparameters["window_stride"]  # Schrittweite der Fensterfuntion
        self.input_size_lstm = hyperparameters["input_size_lstm"]  # Size
        self.criterion_name = hyperparameters["criterion"]
        self.gradient_clip = hyperparameters["gradient_clip"]
        self.gradient_clip_value = hyperparameters["gradient_clip_value"]

        self.device = device
        self.save_model_progress = save_model_progress
        self.debug_mode = debug_mode

    def train(self, model, criterion, optimizer, loss_list, data_loader):
        # Contains losses of a all previously processed Signals
        signal_loss_list = loss_list
        number_signals = 0

        if self.debug_mode:
            debug_index = 0

        # Iterate through number of epochs
        for epoch in range(self.number_epochs):
            # Get Signals from train_loader
            for batch_index, (speech_signal, glottal_flow_derivative_signal) in enumerate(data_loader):
                # Each Batch contains 1x Speech Signal and 1x GlottalFlowDerivativeSignal
                speech_signal = speech_signal[0]  # .to(self.device)
                glottal_flow_derivative_signal = glottal_flow_derivative_signal[0]  # .to(self.device)
                signal_length = speech_signal.size(0)  # Total Signal-Length of Audio-File. (Glottal FLow is the same)

                # If Debugging, print Sizes and Information about Tensors
                if self.debug_mode:
                    if batch_index > 0:
                        break
                    print("Size of Glottal Flow Derivative Signal:" + str(glottal_flow_derivative_signal.size()))
                    print("Size of Speech Signal:" + str(speech_signal.size()))

                sum_sequence_loss = 0  # Will contain the sum of all MSE of the sequences
                number_sequences = 0

                # Windowing the signals into Sequences with Size "Sequence_length" each.
                for slice_start in range(0, signal_length - self.sequence_length + 1, self.window_stride):
                    slice_end = slice_start + self.sequence_length
                    speech_sequence = speech_signal[slice_start:slice_end]
                    glottal_flow_derivative_sequence = glottal_flow_derivative_signal[slice_start:slice_end]

                    # Clears x.grad for every parameter x in the optimizer.
                    optimizer.zero_grad()

                    # Forward-pass
                    # Input has to contain the following
                    # view(): fast and memory efficient reshaping
                    # Input Data: (batch, sequence_length, input_size)
                    input_data = speech_sequence.view(-1, self.sequence_length, self.input_size_lstm)

                    # Feed data into Model. Scores containes the output of the whole network
                    scores = model(input_data)

                    # Criterion calculates loss for every time-step
                    if self.criterion_name == "MSE":
                        # Calculate MSE for whole sequence. Every time step has one error
                        loss_sequence_list = criterion(scores, glottal_flow_derivative_sequence)
                    if self.criterion_name == "COSINE":
                        y = torch.ones(1).to(self.device)  # y is control parameter for the cosine distance error
                        loss_sequence_list = criterion(scores.view(-1, 1),
                                                       glottal_flow_derivative_sequence.view(-1, 1),
                                                       y.view(1).to(self.device))

                    # Backward-Pass
                    # Iterate over all losses of the sequences and backward them through the Network
                    for loss_time_step in enumerate(loss_sequence_list):
                        # Compute gradients. As long gradients are not cleared, they are summed up -> BPTT
                        # retain_graph=True: Not freeing the memory allocated for the graph on the backward pass
                        loss_time_step[1].backward(retain_graph=True)

                    # If gradients are exploding use this function to clip gradients
                    # This is useful for large learning rates!
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_value)

                    # Performs a parameter update based on the calc. gradients
                    optimizer.step()

                    # Calculates the MSE of the whole sequence and accumulate it with the older sequence's MSE
                    sum_sequence_loss += loss_sequence_list.mean().item()  # Loss item returns loss of the sequence

                    number_sequences += 1

                    # If debugging, print sizes and dtypes of Tensors
                    if self.debug_mode:
                        print('Sequence-Index:' + str(number_sequences - 1))
                        print("Tensor-Size of Speech Sequence:" + str(speech_sequence.size()))
                        print(speech_sequence)
                        print(speech_sequence.dtype)
                        print("Tensor-Size of Glottal Flow Derivative:" + str(glottal_flow_derivative_sequence.size()))
                        print(glottal_flow_derivative_sequence)
                        print(glottal_flow_derivative_sequence.dtype)
                        print("Tensor-Size of Scores:" + str(scores.size()))
                        print(scores)
                        print(scores.dtype)
                        print("Tensor-Size of Losses:" + str(loss_sequence_list.size()))
                        print(loss_sequence_list)
                        print('Sequence-MSE: ' + str(loss_sequence_list.mean().item()))
                        if debug_index > 2:
                            break
                        debug_index += 1

                # Calculate average loss over all sequences of the signal and append it to list
                signal_loss_list.append(sum_sequence_loss / number_sequences)

                if self.save_model_progress:
                    # Save Model every 10 files
                    if number_signals % 10 == 9:
                        temp_model_address = self.model_folder_address + self.model_name + '_Signals_' + str(
                            number_signals+1) + '.pth'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': signal_loss_list,
                        }, temp_model_address)

                number_signals += 1

        # Save final Model
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': signal_loss_list,
        }, self.model_address)

        return signal_loss_list
