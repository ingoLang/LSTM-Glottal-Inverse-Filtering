import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, sequence_length: int, stride: int, hyperparameters: dict):
        self.sequence_length = sequence_length
        self.stride = stride  # Deutsch:Schrittweite
        self.sample_rate = hyperparameters["output_sample_rate"]

    def plot_synthetic_signal_sequence(self, speech_signal, glottal_flow_signal, glottal_flow_derivative_signal,
                             predicted_glottal_flow_derivative_signal, sequence_index, save_figure, signal_name = ""):
        speech_signal_numpy = speech_signal.cpu().numpy()
        glottal_flow_signal_numpy = glottal_flow_signal.cpu().numpy()
        glottal_flow_derivative_signal_numpy = glottal_flow_derivative_signal.cpu().numpy()
        predicted_glottal_flow_derivative_signal_numpy = predicted_glottal_flow_derivative_signal.cpu().numpy()

        time = np.linspace(0, (len(speech_signal_numpy) / self.sample_rate) * 1000,
                           num=len(speech_signal_numpy))

        fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(16, 10), constrained_layout=True)

        axes[0].plot(time, speech_signal_numpy, color='black')
        axes[0].set_title('Speech Sequence', size=18)
        axes[0].set_xlabel('Time (Milliseconds)', size=14)
        axes[0].set_ylabel('SpeechSignal', size=14)

        axes[1].plot(time, glottal_flow_signal_numpy, color='black')
        axes[1].set_title('Glottal Flow Sequence', size=18)
        axes[1].set_xlabel('Time (Milliseconds)', size=14)
        axes[1].set_ylabel('Flow', size=14)

        axes[2].plot(time, glottal_flow_derivative_signal_numpy, color='black')
        axes[2].set_title('Glottal Flow Time Derivative Sequence', size=18)
        axes[2].set_xlabel('Time (Milliseconds)', size=14)
        axes[2].set_ylabel('Flow Derivative', size=14)

        axes[3].plot(time, predicted_glottal_flow_derivative_signal_numpy, color='black')
        axes[3].set_title('Predicted Glottal Flow Time Derivative Sequence', size=18)
        axes[3].set_xlabel('Time (Milliseconds)', size=14)
        axes[3].set_ylabel('Flow Derivative', size=14)

        if save_figure:
            fig.savefig(os.path.join("Dataset", "SyntheticSignals_Predictions" + signal_name + "_" + str(sequence_index) + ".png"))

        plt.show()

    def plot_prediction_and_EGGSignal(self, speech_signal, predicted_glottal_flow_derivative_signal,
                                   egg_signal, egg_derivative_signal, sequence_index, save_figure, signal_name = ""):

        speech_signal_numpy = speech_signal.cpu().numpy()
        predicted_glottal_flow_derivative_signal_numpy = predicted_glottal_flow_derivative_signal.cpu().numpy()
        egg_signal_numpy = egg_signal.cpu().numpy()
        egg_derivative_signal = egg_derivative_signal.cpu().numpy()

        time = np.linspace(0, (len(speech_signal_numpy) / self.sample_rate) * 1000,
                           num=len(speech_signal_numpy))

        fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(16, 10), constrained_layout=True)

        axes[0].plot(time, speech_signal_numpy, color='black')
        axes[0].set_title('Speech Sequence', size=18)
        axes[0].set_xlabel('Time (Milliseconds)', size=14)
        axes[0].set_ylabel('SpeechSignal', size=14)

        axes[1].plot(time, egg_signal_numpy, color='black')
        axes[1].set_title('EGG-Sequence', size=18)
        axes[1].set_xlabel('Time (Milliseconds)', size=14)
        axes[1].set_ylabel('Vocal Fold Contract Area', size=14)

        axes[2].plot(time, egg_derivative_signal, color='black')
        axes[2].set_title('EGG Time Derivative Sequence', size=18)
        axes[2].set_xlabel('Time (Milliseconds)', size=14)
        axes[2].set_ylabel('DEGG', size=14)

        axes[3].plot(time, predicted_glottal_flow_derivative_signal_numpy, color='black')
        axes[3].set_title('Predicted Glottal Flow Time Derivative Sequence', size=18)
        axes[3].set_xlabel('Time (Milliseconds)', size=14)
        axes[3].set_ylabel('Flow Derivative', size=14)

        if save_figure:
            fig.savefig(os.path.join("Dataset", "SyntheticSignals_Predictions" + signal_name + "_" + str(sequence_index) + ".png"))

        plt.show()

    # Split Signal into small Sequences: Rect-Window
    def split_signal(self, signal):
        """Splits Speech-Signal and Glottal Flow Derivative-Signal into small Sequences of 14-15ms.
        """
        length = signal.size(0)
        splitted_signal = []
        for slice_start in range(0, length - self.sequence_length + 1, self.stride):
            slice_end = slice_start + self.sequence_length
            splitted_signal.append(signal[slice_start:slice_end])
        # print("Number Sequences:" + str(len(splitted_signal)))

        return splitted_signal

    def visualize_synthetic_signals(self, signals, sequence_index, save_figure, signal_name = ""):
        speech_signal = signals[0]
        glottal_flow_derivative_signal = signals[1]
        glottal_flow_signal = signals[2]
        predicted_glottal_flow_derivative_signal = signals[3]

        splitted_speech_signal = self.split_signal(speech_signal)
        splitted_glottal_flow_signal = self.split_signal(glottal_flow_signal)
        splited_glottal_flow_derivative_signal = self.split_signal(glottal_flow_derivative_signal)
        splited_predicted_glottal_flow_derivative_signal = self.split_signal(predicted_glottal_flow_derivative_signal)

        self.plot_synthetic_signal_sequence(splitted_speech_signal[sequence_index],
                                  splitted_glottal_flow_signal[sequence_index],
                                  splited_glottal_flow_derivative_signal[sequence_index],
                                  splited_predicted_glottal_flow_derivative_signal[sequence_index],
                                  sequence_index,
                                  save_figure,
                                  signal_name
                                  )

    def visualize_prediction_and_EGGsignal(self, signals, sequence_index, save_figure, signal_name = ""):
        speech_signal = signals[0]
        predicted_glottal_flow_derivative_signal = signals[1]
        egg_signal = signals[2]
        egg_derivative_signal = signals[3]

        splitted_speech_signal = self.split_signal(speech_signal)
        splited_predicted_glottal_flow_derivative_signal = self.split_signal(predicted_glottal_flow_derivative_signal)
        splitted_egg_signal = self.split_signal(egg_signal)
        splitted_egg_derivative_signal = self.split_signal(egg_derivative_signal)

        self.plot_prediction_and_EGGSignal(splitted_speech_signal[sequence_index],
                                        splited_predicted_glottal_flow_derivative_signal[sequence_index],
                                        splitted_egg_signal[sequence_index],
                                        splitted_egg_derivative_signal[sequence_index],
                                        sequence_index,
                                        save_figure,
                                        signal_name
                                        )
