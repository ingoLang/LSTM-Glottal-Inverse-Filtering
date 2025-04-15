import os

def get_model_paths(model_name:str):
    model_folder_address = 'Model'
    model_configuration_folder_address = 'ModelConfigurationFiles'
    model_address = os.path.join(model_folder_address, str(model_name + '.pth'))
    model_configuration_file_address = os.path.join(model_folder_address,
                                                    model_configuration_folder_address,
                                                    str(model_name + '.json'))
    return model_address, model_configuration_file_address

def get_signal_paths(signal_name: str):
    dataset_address = 'Dataset'
    dir_address = "_".join(signal_name.split("_")[0:3])
    glottal_flow_name = signal_name + "_GlottalFlow.wav"
    glottal_flow_derivative_name = signal_name + "_GlottalFlowDerivative.wav"
    speech_signal_address = os.path.join(dataset_address, dir_address, signal_name + '.wav')
    glottal_flow_signal_address = os.path.join(dataset_address, dir_address, glottal_flow_name)
    glottal_flow_derivative_signal_address = os.path.join(dataset_address, dir_address, glottal_flow_derivative_name)

    return speech_signal_address, glottal_flow_signal_address, glottal_flow_derivative_signal_address