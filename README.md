# Glottal inverse filtering based on articulatory synthesis and deep learning
We propose a method to estimate the glottal vocal tract excitation from speech signals based on deep learning. To that end, a bidirectional recurrent neural network with long shortterm memory units was trained to predict the glottal airflow derivative from the speech signal. Since natural reference data for this task is unobtainable at the required scale, we used the articulatory speech synthesizer [VocalTractLab](https://www.vocaltractlab.de/) to generate a large dataset containing synchronous connected speech and glottal airflow signals for training. The trained model’s performance was objectively evaluated by means of stationary synthetic signals from the OPENGLOT glottal inverse filtering benchmark dataset and by using our dataset of connected synthetic speech. Compared to the state of the art, the proposed model produced a more accurate estimation using OPENGLOT’s physically synthesized signals but was less accurate for its computationally simulated signals. However, our model was much more accurate and plausible on the connected speech signals, especially for sounds with mixed excitation (e.g. fricatives) or sounds with pronounced zeros in their transfer function (e.g. nasals). 

Authors: Ingo Langheinrich, Simon Stone, Xinyu Zhang, Peter Birkholz

## Features
- Predicts the glottal flow derivative from a speech signal
- The speech signal is processed entirely in time domain
- Useful for speech analysis and voice pathology studies

## Model and Datasets
All 84 trained models were evaluated on the OPENGLOT subset, on a test dataset syntheziced with the VocalTractLab and on the BITS Unit Selection corpus. You can find the best performing models in the repository.

- Training dataset: VocalTractLab dataset for connected speech and glottal flow signals
- Test and validation dataset: We recommend using the BITS dataset with EGG-signals for further validation  
- Some example waveforms are in the dataset folder of the repository

### Preprocessing
- Downsample the speech signal to 8 kHz with sinc-interpolation

### Post processing
- All presented glottal flow signals were obtained by trapezoidal numerical integration of the derivative (code not included in the repo yet)

## Requirements
- Python 3.x (tested under Python 3.10.11)
- PyTorch & Torch-Audio
- NumPy


## License
This project is licensed under the MIT License.