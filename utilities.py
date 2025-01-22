import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def load_dataset(train_path, test_path, label_names, sensors):
    """
    Load training and testing datasets from HDF5 files.

    Parameters
    ----------
    train_path : str
        Path to the training dataset HDF5 file.
    test_path : str
        Path to the testing dataset HDF5 file.
    label_names : list
        List of label names to extract from the HDF5 files.
    sensors : list
        List of sensor names to extract from the HDF5 files.

    Returns
    -------
    tuple
        A tuple containing:
        - X_train_raw (list): List of numpy arrays containing raw training data for each sensor.
        - Y_train_raw (pd.DataFrame): DataFrame containing training labels.
        - X_test (list): List of numpy arrays containing raw testing data for each sensor.
        - Y_test (pd.DataFrame): DataFrame containing testing labels.
    """
    with h5py.File(train_path, 'r') as f_train, h5py.File(test_path, 'r') as f_test:
        # Extract raw training data for each sensor
        X_train_raw = [f_train[sensor][:] for sensor in sensors]
        # Extract and decode training labels
        Y_train_raw = pd.DataFrame(
            [
                [s.decode('utf-8') for s in f_train[label_name][:].flatten()]
                for label_name in label_names
            ]
        ).T
        Y_train_raw.columns = label_names

        # Extract raw testing data for each sensor
        X_test = [f_test[sensor][:] for sensor in sensors]
        # Extract and decode testing labels
        Y_test = pd.DataFrame(
            [
                [s.decode('utf-8') for s in f_test[label_name][:].flatten()]
                for label_name in label_names
            ]
        ).T
        Y_test.columns = label_names

    return X_train_raw, Y_train_raw, X_test, Y_test


class CustomDataset(Dataset):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes
    ----------
    X : list
        List of torch.Tensor objects, where each tensor contains data from a different sensor.
    """

    def __init__(self, X):
        """
        Initialize the CustomDataset with sensor data.

        Parameters
        ----------
        X : list
            List of torch.Tensor objects, where each tensor contains data from a different sensor.
        """
        self.X = X

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset, which is the length of the first sensor's data.
        """
        return len(self.X[0])

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        list
            A list of samples from each sensor at the specified index.
        """
        return [x[idx] for x in self.X]

def freeze_module_parameters(module, freeze=True):
    """
    Helper function to freeze or unfreeze parameters in a PyTorch module.
    """
    for param in module.parameters():
        param.requires_grad = not freeze

def standardize_window(data):
    """
    Standardize the data within each window for each channel.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape (N, C, L), where:
          - N is the number of samples,
          - C is the number of channels,
          - L is the window length.

    Returns
    -------
    numpy.ndarray
        Standardized data (same shape as input).
    """
    N, C, L = data.shape
    mean_ = data.mean(axis=2, keepdims=True)
    std_ = data.std(axis=2, keepdims=True)
    data = (data - mean_) / (std_ + 1e-5)
    return data


def min_max_scale_window(data):
    """
    Apply min-max scaling to the data within each window for each channel.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape (N, C, L), where:
          - N is the number of samples,
          - C is the number of channels,
          - L is the window length.

    Returns
    -------
    numpy.ndarray
        Min-max scaled data (same shape as input).
    """
    N, C, L = data.shape
    min_ = data.min(axis=2, keepdims=True)
    max_ = data.max(axis=2, keepdims=True)
    data = (data - min_) / ((max_ - min_) + 1e-5)
    return data


def standardize(data, mean, std):
    """
    Standardize the data using provided mean and standard deviation.

    Parameters
    ----------
    data : numpy.ndarray
        Input data to be standardized.
    mean : numpy.ndarray
        Mean for standardization.
    std : numpy.ndarray
        Standard deviation for standardization.

    Returns
    -------
    numpy.ndarray
        Standardized data (same shape as input).
    """
    return (data - mean) / (std + 1e-5)


def min_max_scale(data, min_val, max_val):
    """
    Apply min-max scaling to the data using provided min and max values.

    Parameters
    ----------
    data : numpy.ndarray
        Input data to be scaled.
    min_val : numpy.ndarray
        Minimum value for scaling.
    max_val : numpy.ndarray
        Maximum value for scaling.

    Returns
    -------
    numpy.ndarray
        Min-max scaled data (same shape as input).
    """
    return (data - min_val) / ((max_val - min_val) + 1e-5)


def normalize_data(X_train, X_valid, X_test, normalisation):
    """
    Normalize the training, validation, and test datasets using the specified normalization method.

    Available methods:
    - 'std': Standardize across all windows.
    - 'min-max': Min-max scale across all windows.
    - 'std_window': Standardize within each window.
    - 'min-max_window': Min-max scale within each window.

    Parameters
    ----------
    X_train : list
        List of numpy arrays for training data, one per sensor.
    X_valid : list
        List of numpy arrays for validation data, one per sensor.
    X_test : list
        List of numpy arrays for test data, one per sensor.
    normalisation : str
        Normalization method ('std', 'min-max', 'std_window', 'min-max_window').

    Returns
    -------
    tuple
        Normalized training, validation, and test datasets, each as a list of numpy arrays (one per sensor).
    """
    sensor_count = len(X_train)
    channel_counts = [X_train[i].shape[1] for i in range(sensor_count)]

    if normalisation == 'std':
        means_ = [
            X_train[i].mean(axis=(0, 2), keepdims=True)
            for i in range(sensor_count)
        ]
        stds_ = [
            X_train[i].std(axis=(0, 2), keepdims=True)
            for i in range(sensor_count)
        ]

        X_train = [
            standardize(X_train[i], means_[i], stds_[i]) for i in range(sensor_count)
        ]
        X_valid = [
            standardize(X_valid[i], means_[i], stds_[i]) for i in range(sensor_count)
        ]
        X_test = [
            standardize(X_test[i], means_[i], stds_[i]) for i in range(sensor_count)
        ]

    elif normalisation == 'min-max':
        mins_ = [
            X_train[i].min(axis=(0, 2), keepdims=True)
            for i in range(sensor_count)
        ]
        maxs_ = [
            X_train[i].max(axis=(0, 2), keepdims=True)
            for i in range(sensor_count)
        ]

        X_train = [
            min_max_scale(X_train[i], mins_[i], maxs_[i]) for i in range(sensor_count)
        ]
        X_valid = [
            min_max_scale(X_valid[i], mins_[i], maxs_[i]) for i in range(sensor_count)
        ]
        X_test = [
            min_max_scale(X_test[i], mins_[i], maxs_[i]) for i in range(sensor_count)
        ]

    elif normalisation == 'std_window':
        X_train = [standardize_window(X_train[i]) for i in range(sensor_count)]
        X_valid = [standardize_window(X_valid[i]) for i in range(sensor_count)]
        X_test = [standardize_window(X_test[i]) for i in range(sensor_count)]

    elif normalisation == 'min-max_window':
        X_train = [min_max_scale_window(X_train[i]) for i in range(sensor_count)]
        X_valid = [min_max_scale_window(X_valid[i]) for i in range(sensor_count)]
        X_test = [min_max_scale_window(X_test[i]) for i in range(sensor_count)]

    return X_train, X_valid, X_test


def MSE(tensor1, tensor2):
    """
    Calculate the Mean Squared Error (MSE) between two tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first input tensor.
    tensor2 : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        A tensor containing the MSE for each sample in the batch.
    """
    return torch.mean((tensor1 - tensor2) ** 2, dim=1)


def MAE(tensor1, tensor2):
    """
    Calculate the Mean Absolute Error (MAE) between two tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first input tensor.
    tensor2 : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        A tensor containing the MAE for each sample in the batch.
    """
    return torch.mean(torch.abs(tensor1 - tensor2), dim=1)


def MAPE(tensor1, tensor2, epsilon=1e-10):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first input tensor.
    tensor2 : torch.Tensor
        The second input tensor.
    epsilon : float, optional
        A small value added to the denominator to avoid division by zero, by default 1e-10.

    Returns
    -------
    torch.Tensor
        A tensor containing the MAPE for each sample in the batch.
    """
    return torch.mean(torch.abs((tensor1 - tensor2) / (tensor1 + epsilon)) * 100, dim=1)


def get_masked_batch(x_batch_concat, start, end):
    """
    Create a masked batch for a specific segment of the concatenated input batch.

    Parameters
    ----------
    x_batch_concat : torch.Tensor
        Concatenated input batch for all sensors.
    start : int
        Start index of the segment to be masked.
    end : int
        End index of the segment to be masked.

    Returns
    -------
    torch.Tensor
        Masked input batch (same shape as x_batch_concat).
    """
    mask = torch.zeros_like(x_batch_concat)
    mask[:, start:end] = 1
    x_batch_concat_masked = x_batch_concat * mask
    return x_batch_concat_masked


def sensor_specific_loss(criterion, x_batch_concat, x_batch_estimate, WINDOW_LENGTHS, NUM_CHANNELS):
    """
    Calculate the loss for each sensor separately.

    Parameters
    ----------
    criterion : callable
        Loss function to compute the loss (e.g., MSE, MAE, MAPE).
    x_batch_concat : torch.Tensor
        Concatenated input batch for all sensors.
    x_batch_estimate : torch.Tensor
        Concatenated estimated output batch for all sensors.
    WINDOW_LENGTHS : list
        List of window lengths for each sensor.
    NUM_CHANNELS : list
        List of number of channels for each sensor.

    Returns
    -------
    list
        List of loss values for each sensor.
    """
    sensor_loss = []
    start = 0
    for i in range(len(WINDOW_LENGTHS)):
        single_sensor_length = WINDOW_LENGTHS[i] * NUM_CHANNELS[i]
        x_sensor = x_batch_concat[:, start:start+single_sensor_length]
        x_estimated_sensor = x_batch_estimate[:, start:start+single_sensor_length]

        sensor_loss.append(criterion(x_sensor, x_estimated_sensor))
        start += single_sensor_length

    return sensor_loss


def overall_loss(criterion, x_batch_concat, x_batch_estimate):
    """
    Calculate the overall loss between the concatenated input batch and the estimated batch.

    Parameters
    ----------
    criterion : callable
        Loss function to compute the loss (e.g., MSE, MAE, MAPE).
    x_batch_concat : torch.Tensor
        Concatenated input batch for all sensors.
    x_batch_estimate : torch.Tensor
        Concatenated estimated output batch for all sensors.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the overall mean loss.
    """
    return torch.mean(criterion(x_batch_concat, x_batch_estimate))


def get_individual_losses(best_model, SENSORS, WINDOW_LENGTHS, NUM_CHANNELS, x_batch_concat, criterion):
    """
    Compute individual losses for each sensor using the best model.

    Parameters
    ----------
    best_model : torch.nn.Module
        Trained model to estimate the outputs.
    SENSORS : list
        List of sensor names.
    WINDOW_LENGTHS : list
        List of window lengths for each sensor.
    NUM_CHANNELS : list
        List of number of channels for each sensor.
    x_batch_concat : torch.Tensor
        Concatenated input batch for all sensors.
    criterion : callable
        Loss function to compute the loss.

    Returns
    -------
    numpy.ndarray
        Array of individual losses for each sensor in the batch.
    """
    start = 0
    sensor_loss_batch_individual = []

    for i, sensor in enumerate(SENSORS):
        segment_length = WINDOW_LENGTHS[i] * NUM_CHANNELS[i]
        end = start + segment_length

        # Create the masked input for the current sensor
        masked_batch = get_masked_batch(x_batch_concat, start, end)
        _, masked_estimate = best_model(masked_batch)

        # Calculate the loss specifically for the current sensor
        individual_loss = sensor_specific_loss(
            criterion, masked_batch, masked_estimate, WINDOW_LENGTHS, NUM_CHANNELS
        )[i]

        sensor_loss_batch_individual.append(individual_loss)
        start += segment_length

    sensor_loss_batch_individual = torch.stack(sensor_loss_batch_individual).detach().cpu().numpy()
    return sensor_loss_batch_individual


def group_by_segment_id(df, anomaly_score_columns, aggregation_type='mean'):
    """
    Group a DataFrame by 'segment_id' and aggregate the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data to be grouped and aggregated.
    anomaly_score_columns : list
        List of column names containing anomaly scores to be aggregated.
    aggregation_type : str, optional
        Type of aggregation to apply (default is 'mean').

    Returns
    -------
    pd.DataFrame
        DataFrame grouped by 'segment_id' with aggregated values.
    """
    grouping_dict = {
        'combined_label': 'first',
        'split_label': 'first',
        'anomaly_label': 'first',
        'domain_shift_op': 'first',
        'domain_shift_env': 'first'
    }

    for column in anomaly_score_columns:
        grouping_dict[column] = aggregation_type

    grouped_df = df.groupby('segment_id').agg(grouping_dict).reset_index()
    return grouped_df


def calculate_single_auc(df, anomaly_score_column):
    """
    Calculate the Area Under the Curve (AUC) for anomaly detection scores
    by comparing pairwise normal vs. anomalous scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing anomaly scores and labels.
    anomaly_score_column : str
        Column name of the anomaly scores in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the AUC scores for different domains.
    """
    source_mask = df['split_label'].isin(['Normal_Source_Test', 'Anomaly_Source_Test'])
    target_mask = df['split_label'].isin(['Normal_Target_Test', 'Anomaly_Target_Test'])
    normal_samples_mask = df['anomaly_label'] == 'normal'
    anormal_samples_mask = df['anomaly_label'] != 'normal'

    all_normal = df[normal_samples_mask]
    all_anormal = df[anormal_samples_mask]
    source_normal = df[source_mask & normal_samples_mask]
    source_anormal = df[source_mask & anormal_samples_mask]
    target_normal = df[target_mask & normal_samples_mask]
    target_anormal = df[target_mask & anormal_samples_mask]

    domain_names = ['all', 'source', 'target']
    domains = [
        (all_normal, all_anormal),
        (source_normal, source_anormal),
        (target_normal, target_anormal)
    ]

    AUC = {}
    for k, (normal_df, anormal_df) in enumerate(domains):
        AUC_mtx = np.zeros((normal_df.shape[0], anormal_df.shape[0]))
        for i, normal_val in enumerate(normal_df[anomaly_score_column]):
            for j, anormal_val in enumerate(anormal_df[anomaly_score_column]):
                AUC_mtx[i, j] = int(normal_val < anormal_val)

        AUC_score = AUC_mtx.sum() / (AUC_mtx.shape[0] * AUC_mtx.shape[1])
        AUC[domain_names[k]] = AUC_score

    return pd.DataFrame(AUC, index=[anomaly_score_column])


def initialise_dataloaders(PARAMS):
    """
    Initialize DataLoaders for training, validation, and testing.

    This function:
      1. Loads the HDF5 datasets.
      2. Splits training data into train and validation sets.
      3. Normalizes data according to the chosen normalization method.
      4. Constructs PyTorch DataLoaders.

    Parameters
    ----------
    PARAMS : dict
        Dictionary of parameters containing:
        - 'machine' (str): Name of the machine (used in dataset paths).
        - 'sensors' (list): Names of sensors to load.
        - 'seed' (int): Random seed for reproducibility.
        - 'device' (str): Device type ('cpu', 'cuda', 'mps', etc.).
        - 'valid_size' (float): Fraction of training data used for validation.
        - 'batch_size' (int): Batch size for DataLoaders.
        - 'normalisation' (str): Normalization method ('std', 'min-max', 'std_window', 'min-max_window').

    Returns
    -------
    tuple
        A tuple containing:
        - train_data_loader (DataLoader): DataLoader for the training set.
        - valid_data_loader (DataLoader): DataLoader for the validation set.
        - test_data_loader (DataLoader): DataLoader for the test set.
        - Y_test (pd.DataFrame): DataFrame of test labels.
        - NUM_CHANNELS (dict): Dictionary of sensor -> number of channels.
        - WINDOW_LENGTHS (dict): Dictionary of sensor -> window length.
    """
    TRAIN_DATASET_PATH = f'data/{PARAMS["machine"]}/windowed/train_dataset_window_0.100s.h5'
    TEST_DATASET_PATH = f'data/{PARAMS["machine"]}/windowed/test_dataset_window_0.100s.h5'

    LABEL_NAMES = [
        'segment_id',
        'split_label',
        'anomaly_label',
        'domain_shift_op',
        'domain_shift_env'
    ]

    # Load the dataset
    X_train_raw, Y_train_raw, X_test, Y_test = load_dataset(
        TRAIN_DATASET_PATH, TEST_DATASET_PATH, LABEL_NAMES, PARAMS["sensors"]
    )

    # Set random seeds for reproducibility
    torch.manual_seed(PARAMS['seed'])
    if PARAMS['device'] == 'mps':
        torch.mps.manual_seed(PARAMS['seed'])
    elif PARAMS['device'] == 'cuda':
        torch.cuda.manual_seed(PARAMS['seed'])
    elif PARAMS['device'] == 'cpu':
        torch.manual_seed(PARAMS['seed'])
    else:
        raise ValueError(f"Unsupported device type: {PARAMS['device']}")

    # Combine anomaly labels and domain shift labels into a single label
    Y_train_raw['combined_label'] = (
        Y_train_raw['anomaly_label']
        + Y_train_raw['domain_shift_op']
        + Y_train_raw['domain_shift_env']
    )
    Y_test['combined_label'] = (
        Y_test['anomaly_label']
        + Y_test['domain_shift_op']
        + Y_test['domain_shift_env']
    )

    # Stratified split into training and validation sets
    train_indices, valid_indices, _, _ = train_test_split(
        range(len(Y_train_raw)),
        Y_train_raw,
        stratify=Y_train_raw['combined_label'],
        test_size=PARAMS['valid_size'],
        random_state=PARAMS['seed']
    )

    X_train = [sensor_data[train_indices] for sensor_data in X_train_raw]
    X_valid = [sensor_data[valid_indices] for sensor_data in X_train_raw]
    Y_train = Y_train_raw.iloc[train_indices].reset_index(drop=True)
    Y_valid = Y_train_raw.iloc[valid_indices].reset_index(drop=True)

    # Normalize the datasets
    X_train, X_valid, X_test = normalize_data(
        X_train, X_valid, X_test, PARAMS['normalisation']
    )

    # Extract the number of channels and window lengths for each sensor
    NUM_CHANNELS = {
        PARAMS["sensors"][i]: x.shape[1] for i, x in enumerate(X_train)
    }
    WINDOW_LENGTHS = {
        PARAMS["sensors"][i]: x.shape[2] for i, x in enumerate(X_train)
    }

    # Convert to torch.Tensor
    X_train_tensor = [torch.from_numpy(x) for x in X_train]
    X_valid_tensor = [torch.from_numpy(x) for x in X_valid]
    X_test_tensor = [torch.from_numpy(x) for x in X_test]

    # Create Dataset objects
    train_dataset = CustomDataset(X_train_tensor)
    valid_dataset = CustomDataset(X_valid_tensor)
    test_dataset = CustomDataset(X_test_tensor)

    # Create DataLoaders
    train_data_loader = DataLoader(
        train_dataset, batch_size=PARAMS['batch_size'], shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=PARAMS['batch_size'], shuffle=False
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=PARAMS['batch_size'], shuffle=False
    )

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        Y_test,
        NUM_CHANNELS,
        WINDOW_LENGTHS
    )