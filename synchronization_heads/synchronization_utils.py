import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import datetime


def conv_layer_output_size(input_size, padding, dilation, kernel_size, stride):
    """
    Calculate the output size of a standard convolutional layer.
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv_transpose_output_size(input_size, padding, dilation, kernel_size, stride):
    """
    Calculate the output size of a transposed convolutional layer.
    """
    return stride * (input_size - 1) + dilation * (kernel_size - 1) - 2 * padding + 1


def calculate_layer_outputs(combinations):
    """
    Calculates the layer outputs for all combinations provided.

    Parameters:
        combinations (pd.DataFrame): DataFrame containing configurations for each combination.
        input_size (int): The input size for the first layer of all combinations.

    Returns:
        pd.DataFrame: A DataFrame with calculated outputs for valid combinations.
    """
    # Helper functions for calculating layer outputs
    def calculate_output_size(layer_type, input_size, padding, dilation, kernel_size, stride):
        if layer_type == 'conv':
            return conv_layer_output_size(input_size, padding, dilation, kernel_size, stride)
        elif layer_type == 'conv_transpose':
            return conv_transpose_output_size(input_size, padding, dilation, kernel_size, stride)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    filters = {
        'conv-conv': (combinations['type_1'] == 'conv') & (combinations['type_2'] == 'conv'),
        'conv-conv_transpose': (combinations['type_1'] == 'conv') & (combinations['type_2'] == 'conv_transpose'),
        'conv_transpose-conv': (combinations['type_1'] == 'conv_transpose') & (combinations['type_2'] == 'conv'),
        'conv_transpose-conv_transpose': (combinations['type_1'] == 'conv_transpose') & (combinations['type_2'] == 'conv_transpose'),
    }

    results = []

    # Process each type pair
    for pair_name, filter_condition in filters.items():
        # Avoid SettingWithCopyWarning
        filtered_combinations = combinations[filter_condition].copy()
        layer_1_type, layer_2_type = pair_name.split('-')

        # Compute outputs for each layer
        filtered_combinations['input_1'] = calculate_output_size(
            layer_1_type,
            filtered_combinations['input_size'],
            filtered_combinations['padding_1'],
            filtered_combinations['dilation_1'],
            filtered_combinations['kernel_size_1'],
            filtered_combinations['stride_1']
        )

        filtered_combinations['input_2'] = calculate_output_size(
            layer_2_type,
            filtered_combinations['input_1'],
            filtered_combinations['padding_2'],
            filtered_combinations['dilation_2'],
            filtered_combinations['kernel_size_2'],
            filtered_combinations['stride_2']
        )

        filtered_combinations['output_1'] = calculate_output_size(
            'conv' if layer_2_type == 'conv_transpose' else 'conv_transpose',
            filtered_combinations['input_2'],
            filtered_combinations['padding_2'],
            filtered_combinations['dilation_2'],
            filtered_combinations['kernel_size_2'],
            filtered_combinations['stride_2']
        )

        filtered_combinations['output_2'] = calculate_output_size(
            'conv' if layer_1_type == 'conv_transpose' else 'conv_transpose',
            filtered_combinations['output_1'],
            filtered_combinations['padding_1'],
            filtered_combinations['dilation_1'],
            filtered_combinations['kernel_size_1'],
            filtered_combinations['stride_1']
        )

        results.append(filtered_combinations)

    # Concatenate and filter valid results
    results = pd.concat(results, ignore_index=True)
    return results


def filter_combinations(combinations, constraints):
    """
    Filters the combinations DataFrame based on constraints.

    Parameters:
        combinations (pd.DataFrame): DataFrame of layer combinations.
        constraints (list): List of constraints to apply (string expressions).

    Returns:
        pd.DataFrame: Filtered combinations.
    """
    # Precompute required values to handle constraints involving // operations
    combinations = combinations.copy()
    combinations['kernel_size_1_half'] = combinations['kernel_size_1'] // 2
    combinations['kernel_size_2_half'] = combinations['kernel_size_2'] // 2

    # Update constraints to use precomputed values
    updated_constraints = [
        constraint.replace("kernel_size_1 // 2", "kernel_size_1_half")
                  .replace("kernel_size_2 // 2", "kernel_size_2_half")
        for constraint in constraints
    ]

    # Apply constraints using query
    for condition in updated_constraints:
        combinations = combinations.query(condition)

    # Drop auxiliary columns after filtering
    combinations = combinations.drop(
        columns=['kernel_size_1_half', 'kernel_size_2_half'])

    return combinations


def generate_combinations(solution_space):
    arrays = [np.array(values) for values in solution_space.values()]
    cartesian_product = np.stack(np.meshgrid(
        *arrays), -1).reshape(-1, len(arrays))
    out = pd.DataFrame(cartesian_product, columns=solution_space.keys())
    # Map integers back to strings
    out['type_1'] = out['type_1'].map({0: 'conv', 1: 'conv_transpose'})
    out['type_2'] = out['type_2'].map({0: 'conv', 1: 'conv_transpose'})
    return out


def create_synchronization_head(input_sensor_channels, output_sensor_channels, groups, parameters, type='input'):
    """
    Creates a two-layer convolution-based model (either Conv1d or ConvTranspose1d) with batch normalization 
    and ReLU activation after the first layer. If 'type' is 'input', it also applies batch normalization 
    and ReLU after the second layer.

    The architecture:
        Layer 1: Conv1d or ConvTranspose1d
        BatchNorm1d
        ReLU
        Layer 2: Conv1d or ConvTranspose1d
        (If type='input'):
            BatchNorm1d
            ReLU

    Parameters:
        input_sensor_channels (int): Number of input channels from the input sensor.
        output_sensor_channels (int): Number of output channels to produce.
        parameters (pd.Series or dict): Should contain keys:
            type_1, kernel_size_1, stride_1, padding_1, dilation_1
            type_2, kernel_size_2, stride_2, padding_2, dilation_2
            Each type_* can be either 'conv' or 'conv_transpose'.

        type (str): 
            - 'input': The model transforms input_sensor_channels -> output_sensor_channels
            - 'output': The model transforms output_sensor_channels -> input_sensor_channels

        fuse_channels (bool): 
            If True, use groups=1 (all channels fused).
            If False, use groups=input_sensor_channels (grouped convolution).

    Returns:
        nn.Sequential: A sequential model composed of:
            - Conv/ConvTranspose layer
            - BatchNorm1d
            - ReLU
            - Conv/ConvTranspose layer
            - If type='input':
                - BatchNorm1d
                - ReLU
    """

    # Extract parameters for the first layer
    layer_1_type = parameters['type_1']
    kernel_size_1 = parameters['kernel_size_1']
    stride_1 = parameters['stride_1']
    padding_1 = parameters['padding_1']
    dilation_1 = parameters['dilation_1']

    layer_2_type = parameters['type_2']
    kernel_size_2 = parameters['kernel_size_2']
    stride_2 = parameters['stride_2']
    padding_2 = parameters['padding_2']
    dilation_2 = parameters['dilation_2']

    # Create the first layer
    if layer_1_type == 'conv':
        layer_1 = nn.Conv1d(
            in_channels=input_sensor_channels,
            out_channels=output_sensor_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
            dilation=dilation_1,
            groups=groups
        )
    else:
        layer_1 = nn.ConvTranspose1d(
            in_channels=input_sensor_channels,
            out_channels=output_sensor_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
            dilation=dilation_1,
            groups=groups
        )

    # Batch normalization and ReLU after first layer

    relu1 = nn.ReLU()
    bn1 = nn.BatchNorm1d(num_features=output_sensor_channels)

    # Create the second layer
    if layer_2_type == 'conv':
        layer_2 = nn.Conv1d(
            in_channels=output_sensor_channels,
            out_channels=output_sensor_channels,
            kernel_size=kernel_size_2,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=groups
        )
    else:
        layer_2 = nn.ConvTranspose1d(
            in_channels=output_sensor_channels,
            out_channels=output_sensor_channels,
            kernel_size=kernel_size_2,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=groups
        )

    # Build the layers into a list first
    layers = [layer_1, relu1, bn1, layer_2]

    # If type='input', add BatchNorm and ReLU after the second layer as well
    if type == 'input':
        relu2 = nn.ReLU()
        bn2 = nn.BatchNorm1d(num_features=output_sensor_channels)
        layers.extend([relu2, bn2])

    # Return the final sequential model
    return nn.Sequential(*layers)


def invert_synchronization_head_parameters(parameters):
    """
    Inverts the parameters of a synchronization head.

    Parameters:
        parameters (dict): The parameters of the synchronization head.

    Returns:
        dict: The inverted parameters.
    """
    out_parameters = parameters.copy()
    out_parameters['input_size'] = parameters['input_2']

    # Swap 'conv' and 'conv_transpose' in type_1
    out_parameters['type_1'] = 'conv' if parameters['type_1'] == 'conv_transpose' else 'conv_transpose'
    out_parameters['kernel_size_1'] = parameters['kernel_size_2']
    out_parameters['stride_1'] = parameters['stride_2']
    out_parameters['padding_1'] = parameters['padding_2']
    out_parameters['dilation_1'] = parameters['dilation_2']

    out_parameters['type_2'] = 'conv' if parameters['type_2'] == 'conv_transpose' else 'conv_transpose'
    out_parameters['kernel_size_2'] = parameters['kernel_size_1']
    out_parameters['stride_2'] = parameters['stride_1']
    out_parameters['padding_2'] = parameters['padding_1']
    out_parameters['dilation_2'] = parameters['dilation_1']

    return out_parameters


def get_optimum_window_length(WINDOW_LENGTHS, NUM_CHANNELS, lambda_):
    # Convert window lengths to numpy array
    window_lengths = np.array(list(WINDOW_LENGTHS.values()))
    # Convert number of channels to numpy array
    num_channels = np.array(list(NUM_CHANNELS.values()))
    # Weighted average window length
    average_window_length = np.mean(window_lengths * num_channels)
    optimum_window_length = average_window_length * lambda_  # Optimum window length
    return optimum_window_length, average_window_length  # Return the calculated values


def get_possible_optimum_window_length(WINDOW_LENGTHS, NUM_CHANNELS, lambda_, common_window_lengths):
    optimum_window_length, average_window_length = get_optimum_window_length(
        # Get optimum and average window lengths
        WINDOW_LENGTHS, NUM_CHANNELS, lambda_)
    # Distances to common window lengths
    distances = np.abs(np.array(common_window_lengths) - optimum_window_length)
    best_possible_window_length = common_window_lengths[np.argmin(
        distances)]  # Closest common window length
    possible_lambda = best_possible_window_length / \
        average_window_length  # Corresponding lambda value
    # Return the best window length and lambda
    return best_possible_window_length, possible_lambda


def simulate_synchronization_head(SENSORS, sensor, num_simulation, C_sync, train_data_loader, candidate_parameters_sensor_dict, syncron_window_length, NUM_CHANNELS, PARAMS):
    dataloader_iter = iter(train_data_loader)
    batch = next(dataloader_iter)
    candidate_parameters_sensor = candidate_parameters_sensor_dict[sensor]
    window_length_mask = candidate_parameters_sensor['input_2'] == syncron_window_length
    candidate_parameters_sensor = candidate_parameters_sensor[window_length_mask].reset_index(
        drop=True)

    sensor_index = np.where(np.array(SENSORS) == sensor)[0][0]

    simulation_durations = []
    for j in range(len(candidate_parameters_sensor)):
        input = batch[sensor_index].to(PARAMS['device'])
        parameters = candidate_parameters_sensor.iloc[j]
        synchronization_head_input = create_synchronization_head(
            NUM_CHANNELS[sensor], C_sync, parameters, type='input').to(PARAMS['device'])
        out_parameters = invert_synchronization_head_parameters(parameters)
        synchronization_head_output = create_synchronization_head(
            C_sync, NUM_CHANNELS[sensor], out_parameters, type='output').to(PARAMS['device'])
        synchronization_head_input.eval()
        synchronization_head_output.eval()

        torch.mps.synchronize()
        start = datetime.datetime.now()
        with torch.no_grad():
            for i in range(num_simulation):
                output = synchronization_head_output(
                    synchronization_head_input(input))
        torch.mps.synchronize()
        end = datetime.datetime.now()
        simulation_durations.append(
            (end - start).total_seconds()/num_simulation)

    simulation_durations = np.array(simulation_durations)
    candidate_parameters_sensor = candidate_parameters_sensor.copy()
    candidate_parameters_sensor['simulation_durations'] = simulation_durations

    return candidate_parameters_sensor


def put_number_of_operation(sensor, candidate_parameters_sensor, C_network, C_sync, NUM_CHANNELS):
    out_candidate_parameters_sensor = candidate_parameters_sensor.copy()
    C_in = NUM_CHANNELS[sensor]
    layer_1_operation = C_in * C_sync * \
        candidate_parameters_sensor['input_1'] * \
        candidate_parameters_sensor['kernel_size_1']
    layer_2_operation = C_sync * C_network * \
        candidate_parameters_sensor['input_2'] * \
        candidate_parameters_sensor['kernel_size_2']
    layer_3_operation = C_network * C_sync * \
        candidate_parameters_sensor['output_1'] * \
        candidate_parameters_sensor['kernel_size_2']
    layer_4_operation = C_sync * C_in * \
        candidate_parameters_sensor['output_2'] * \
        candidate_parameters_sensor['kernel_size_1']

    total_operation = layer_1_operation + layer_2_operation + \
        layer_3_operation + layer_4_operation
    out_candidate_parameters_sensor['total_operation'] = total_operation
    return out_candidate_parameters_sensor


def find_combinations_for_sensors(SENSORS, WINDOW_LENGTHS, combinations):
    """
    Find the combinations for each possible input window length and fill dictionary for each sensor.

    Parameters:
        SENSORS (list): List of sensor names.
        WINDOW_LENGTHS (dict): Dictionary of window lengths for each sensor.
        combinations (pd.DataFrame): DataFrame containing layer combinations.

    Returns:
        dict: Dictionary containing combinations for each sensor.
    """
    window_combinations_dict = {}
    for sensor in SENSORS:
        if WINDOW_LENGTHS[sensor] in window_combinations_dict.keys():
            print(f"Skipping {
                  sensor} as it has the same window length as another")
            continue
        window_combinations = combinations.copy()
        window_combinations['input_size'] = WINDOW_LENGTHS[sensor]

        window_combinations = calculate_layer_outputs(
            window_combinations).reset_index(drop=True)

        same_input_output_filter = window_combinations['input_size'] == window_combinations['output_2'].reset_index(
            drop=True)
        window_combinations = window_combinations[same_input_output_filter].reset_index(
            drop=True)

        window_combinations_dict[WINDOW_LENGTHS[sensor]] = window_combinations

    sensor_combinations_dict = {
        sensor: window_combinations_dict[WINDOW_LENGTHS[sensor]] for sensor in SENSORS}
    return sensor_combinations_dict


def find_common_window_lengths_and_filter(SENSORS, sensor_combinations_dict):
    """
    Find common window lengths across all sensors and filter the combinations.

    Parameters:
        SENSORS (list): List of sensor names.
        sensor_combinations_dict (dict): Dictionary containing combinations for each sensor.

    Returns:
        list: List of common window lengths.
        dict: Dictionary containing filtered combinations for each sensor.
    """
    # Find common window lengths across all sensors
    common_window_lengths = set(
        sensor_combinations_dict[SENSORS[0]]['input_2'])
    for sensor in SENSORS[1:]:
        common_window_lengths = common_window_lengths.intersection(
            set(sensor_combinations_dict[sensor]['input_2']))
    common_window_lengths = list(common_window_lengths)

    # Filter out non-common window lengths for each sensor
    candidate_parameters_sensor_dict = {}
    for sensor in SENSORS:
        candidate_parameters_sensor_dict[sensor] = sensor_combinations_dict[sensor][sensor_combinations_dict[sensor]['input_2'].isin(
            common_window_lengths)]

    return common_window_lengths, candidate_parameters_sensor_dict


def find_optimum_window_length_and_solutions(WINDOW_LENGTHS, NUM_CHANNELS, lambda_, common_window_lengths, SENSORS, candidate_parameters_sensor_dict):
    """
    Find the optimum window length and corresponding lambda value, and print the number of possible solutions for each sensor.

    Parameters:
        WINDOW_LENGTHS (dict): Dictionary of window lengths for each sensor.
        NUM_CHANNELS (dict): Dictionary of number of channels for each sensor.
        lambda_ (float): Lambda parameter.
        common_window_lengths (list): List of common window lengths.
        SENSORS (list): List of sensor names.
        candidate_parameters_sensor_dict (dict): Dictionary containing combinations for each sensor.

    Returns:
        int: Optimum window length.
        float: Corresponding lambda value.
    """
    # Get the best possible window length and the corresponding lambda value
    syncron_window_length, possible_lambda = get_possible_optimum_window_length(
        WINDOW_LENGTHS, NUM_CHANNELS, lambda_, common_window_lengths)
    print(f"Optimum window length: {
          syncron_window_length}, corresponding lambda: {possible_lambda}")

    # Print number of possible solutions for each sensor
    for sensor in SENSORS:
        candidate_parameters_sensor = candidate_parameters_sensor_dict[sensor]
        candidate_parameters_sensor = candidate_parameters_sensor[
            candidate_parameters_sensor['input_2'] == syncron_window_length]
        print(f"For sensor {sensor} there are {
              candidate_parameters_sensor.shape[0]} solutions")

    return syncron_window_length, possible_lambda


def obtain_best_parameters_for_each_sensor(SENSORS, candidate_parameters_sensor_dict, syncron_window_length, C_network, C_sync, NUM_CHANNELS):
    """
    Obtain the best parameters for each sensor.

    Parameters:
      SENSORS (list): List of sensor names.
      candidate_parameters_sensor_dict (dict): Dictionary containing combinations for each sensor.
      syncron_window_length (int): The synchronization window length.
      C_network (int): Number of channels in the network.
      C_sync (int): Number of synchronization channels.
      NUM_CHANNELS (dict): Dictionary of number of channels for each sensor.

    Returns:
      dict: Dictionary containing the best parameters for each sensor.
    """
    sensor_syncron_parameters = {}
    for sensor in SENSORS:
        candidate_parameters_sensor = candidate_parameters_sensor_dict[sensor]
        candidate_parameters_sensor = candidate_parameters_sensor[
            candidate_parameters_sensor['input_2'] == syncron_window_length]

        candidate_parameters_sensor = put_number_of_operation(sensor,
                                                              candidate_parameters_sensor,
                                                              C_network,
                                                              C_sync,
                                                              NUM_CHANNELS)

        best_parameters = candidate_parameters_sensor.iloc[np.argmin(
            candidate_parameters_sensor['total_operation'])]
        sensor_syncron_parameters[sensor] = best_parameters

    return sensor_syncron_parameters


def initialize_parameters(SENSORS, WINDOW_LENGTHS, NUM_CHANNELS, C_sync=16, C_network=16, lambda_=0):
    solution_space = {
        "kernel_size_1": np.arange(1, 16),
        "stride_1": [1, 2, 3, 4],
        "padding_1": np.arange(0, 8),
        "dilation_1": [1, 2, 3],
        "type_1": [0, 1],  # 0 for "conv", 1 for "conv_transpose"

        "kernel_size_2": np.arange(1, 16),
        "stride_2": [1, 2, 3, 4],
        "padding_2": np.arange(0, 8),
        "dilation_2": [1, 2, 3],
        "type_2": [0, 1],  # 0 for "conv", 1 for "conv_transpose"
    }

    constraints = [
        "kernel_size_1 // 2 >= padding_1",
        "kernel_size_2 // 2 >= padding_2",
        "kernel_size_1 // 2 >= dilation_1",
        "kernel_size_2 // 2 >= dilation_2",
        "kernel_size_1 // 2 >= stride_1",
        "kernel_size_2 // 2 >= stride_2",
        "type_1 == type_2",
        "kernel_size_1 > stride_1 + dilation_1",
    ]

    # Generate all possible combinations of the solution space
    combinations = generate_combinations(solution_space)

    # Filter the combinations based on the given constraints
    combinations = filter_combinations(combinations, constraints)

    # Find the combinations for each sensor
    sensor_combinations_dict = find_combinations_for_sensors(
        SENSORS, WINDOW_LENGTHS, combinations)

    # Find common window lengths across all sensors and filter the combinations
    common_window_lengths, candidate_parameters_sensor_dict = find_common_window_lengths_and_filter(
        SENSORS, sensor_combinations_dict)

    # Find the optimum window length and corresponding lambda value
    syncron_window_length, possible_lambda = find_optimum_window_length_and_solutions(
        WINDOW_LENGTHS, NUM_CHANNELS, lambda_, common_window_lengths, SENSORS, candidate_parameters_sensor_dict)

    # Obtain the best parameters for each sensor
    sensor_syncron_parameters = obtain_best_parameters_for_each_sensor(
        SENSORS, candidate_parameters_sensor_dict, syncron_window_length, C_network, C_sync, NUM_CHANNELS)

    return sensor_syncron_parameters


class create_fc_head(nn.Module):
    """
    A fully-connected (FC) head that operates channel by channel.

    This can be used for:
      - Synchronization: map from a sensor's raw length L_sensor to a common length L_common.
      - Projection: map from the common length L_common back to L_sensor.
    
    Shape:
        - Input:  (N, C, L_in)
        - Output: (N, C, L_out)
    """

    def __init__(self, input_size, output_size, num_channels, num_layers):
        """
        Args:
            input_size (int): L_in
            output_size (int): L_out
            num_channels (int): C
            num_layers (int): number of FC layers per channel
        """
        super(create_fc_head, self).__init__()
        self.fc_stacks = nn.ModuleList()

        # Create channel-specific FC stacks
        for _ in range(num_channels):
            layers = []
            current_size = input_size
            for layer_idx in range(num_layers):
                # FC layer
                fc = nn.Linear(current_size, output_size)
                layers.append(fc)

                # Add ReLU + BatchNorm except after the last layer
                if layer_idx < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(output_size))

                current_size = output_size

            # Wrap in a Sequential
            self.fc_stacks.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, L_in)
        
        Returns:
            torch.Tensor: (N, C, L_out)
        """
        outputs = []
        for c in range(x.size(1)):
            # Channel slice => (N, L_in)
            channel_input = x[:, c, :]
            channel_output = self.fc_stacks[c](channel_input)  # (N, L_out)
            outputs.append(channel_output.unsqueeze(1))        # (N, 1, L_out)

        # Concat the per-channel outputs => (N, C, L_out)
        return torch.cat(outputs, dim=1)
