import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from scipy import signal
from scipy.signal import filtfilt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import mode


def movingstd(data, window_size, windowmode='central'):
    """
 movingstd: efficient windowed standard deviation of a time series
 usage: data_std = movingstd(data,window_size,windowmode)

 Movingstd uses filter to compute the standard deviation, using
 the trick of std = sqrt((sum(x.^2) - n*xbar.^2)/(n-1)).
 Beware that this formula can suffer from numerical problems for
 data which is large in magnitude. Your data is automatically
 centered and scaled to alleviate these problems.

 At the ends of the series, when filter would generate spurious
 results otherwise, the standard deviations are corrected by
 the use of shorter window lengths.

    :param  data: numpy array containing time series data
    :param window_size : size of the moving window to use (see windowmode)
    All windowmodes adjust the window width near the ends of
    the series as necessary. window_size must be an integer, at least 1 for a 'central' window,
    and at least 2 for 'forward' or 'backward'

    :param windowmode: (OPTIONAL) flag, denotes the type of moving window used
    DEFAULT: 'central'
    windowmode = 'central' --> use a sliding window centered on each point in the series.
    The window will have total width of 2*window_size+1 points, thus k points on each side.
    windowmode = 'backward' --> use a sliding window that uses the current point and looks back over a total of
    window_size points.
    windowmode = 'forward' --> use a sliding window that uses the current point and looks forward over a total of
    window_size points.

    Any simple contraction of the above options is valid, even as short as a single character 'c', 'b', or 'f'. Case is
    ignored.

    :return s * data_std - vector containing the windowed standard deviation. length(data_std) == length(data.shape[0])
    """

    # Check for valid windowmode
    valid = ['central', 'forward', 'backward']
    if not isinstance(windowmode, str) or windowmode.lower() not in valid:
        raise ValueError("Windowmode must be a character flag, matching the allowed modes: 'c', 'b', or 'f'.")
    windowmode = windowmode.lower()

    # Check for valid k
    n = len(data)
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(
            "window_size must be an integer, at least 1 for a 'central' window, and at least 2 for 'forward' or 'backward'.")
    if windowmode == 'central' and window_size < 1:
        raise ValueError("window_size must be at least 1 for windowmode = 'central'.")
    if (windowmode != 'central' and window_size < 2) or n < window_size:
        raise ValueError("window_size is too large for this short of a series.")

    # Compute the data's magnitudes if there are more than 1 axis
    data = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)
    # Center and scale the data to improve numerical analysis
    data = data - np.mean(data)
    data_std = np.std(data)
    data = data / data_std

    # Compute squared elements
    data2 = data ** 2

    # Compute moving standard deviation
    if windowmode == 'central':
        B = np.ones(2 * window_size + 1)
        s = np.sqrt(
            (np.convolve(data2, B, mode='same') - np.convolve(data, B, mode='same') ** 2 * (1 / (2 * window_size + 1)))
            / (2 * window_size))
        # s[k:(n - k)] = s[(2 * k):]
    elif windowmode == 'forward':
        B = np.ones(window_size)
        s = np.sqrt((np.convolve(data2, B, mode='same') - np.convolve(data, B, mode='same') ** 2 * (1 / window_size)) /
                    (window_size - 1))
        # s[:(n - k)] = s[k:]
    elif windowmode == 'backward':
        B = np.ones(window_size)
        s = np.sqrt((np.convolve(data2, B, mode='same') - np.convolve(data, B, mode='same') ** 2 * (1 / window_size)) /
                    (window_size - 1))

    return s * data_std # Scale the std to be with the same scale of the original data

def bandpass_filter(data, low_cut, high_cut, sampling_rate, order):
    """Apply a band-pass filter to the input data.
    :param data: NumPy array of shape (n_points, 3)
    :param low_cut: Lower frequency cutoff (Hz)
    :param high_cut: Upper frequency cutoff (Hz)
    :param sampling_rate: Sampling frequency (Hz)
    :param order: Order of the Butterworth filter

    :return: Filtered data
    """
    nyq = 0.5 * sampling_rate
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def resample(data, labels, chorea, original_fs, target_fs):
    '''
    :param x: Numpy array. Data to resample.
    :param original_fs: Float, the raw data sampling rate
    :param target_fs: Float, the sampling rate of the resampled signal
    :return: resampled data
    '''
    # calculate resampling factor
    resampling_factor = original_fs / target_fs
    # calculate number of samples in the resampled data and labels
    num_samples = int(len(data) / resampling_factor)
    # use scipy.signal.resample function to resample data, labels, and subjects
    resampled_data = signal.resample(data, num_samples)
    # Resample the labels to match the new length of the resampled data
    resampled_labels = signal.resample(labels.astype(float), num_samples)
    resampled_labels = np.round(resampled_labels).astype(int)
    resampled_chorea = signal.resample(chorea.astype(float), num_samples)
    resampled_chorea = np.round(resampled_chorea).astype(int)

    return resampled_data, resampled_labels, resampled_chorea

def data_windowing(data, labels, chorea, window_size, window_overlap, std_idx):
    """
    Dividing the data into fixed-time windows

    Parameters:
    :param acc_np: NumPy array of shape (n_points, 3)
    :param labels_np: NumPy array of shape (n_points, 1)
    :param window_size: Integer
    :param window_overlap: overlap between windows (in samples)
    :param std_idx: Numpy array of shape (n_points,1) indicating removed low-activity samples
    :TODO ADD param walk_thr: Float, the percent of walking in window to consider it as gait (label=1)

    :return: Data and labels divided into time-fixed windows
    """

    # Create an empty array to hold the windowed data
    windowed_data_all = np.empty((0, 3, window_size))
    windowed_labels_all = np.empty((0, 1))
    windowed_chorea_all = np.empty((0, 1))

    # Number of seconds that are not overlap between neighboring windows
    non_overlap = window_size - window_overlap
    # Create vector that indicate which part of the data was removed in the division into windows
    inclusion_idx = std_idx[std_idx == 1]

    # The upper limit indicate the index of the last sample that can be divided into windows
    # (e.g., if the step of the sliding windows is 100 and data.shape[0]=607, the upper limit is 600)
    upper_limit = data.shape[0] // non_overlap * non_overlap
    # Remove the end of the signal
    inclusion_idx[upper_limit - window_overlap:] = 0

    # Use sliding windows to divide the data
    windowed_data = sliding_window_view(data, window_size, 0)[::non_overlap, :]
    windowed_labels = sliding_window_view(labels, window_size, 0)[::non_overlap].squeeze()
    windowed_chorea = sliding_window_view(chorea, window_size, 0)[::non_overlap].squeeze()
    # Save the indices that indicating which windows belong to which subject
    NumWin = windowed_labels.shape[0]
    # Assign the mode label as the label for each window
    chorea_valid_samples = np.sum(windowed_chorea>=0, axis=1)
    windowed_chorea_sum = np.sum(windowed_chorea*(windowed_chorea>=0), axis=1)
    windowed_chorea = [windowed_chorea_sum[i]/chorea_valid_samples[i] if chorea_valid_samples[i] > windowed_data.shape[-1]/2 else -1 for i in range(len(windowed_chorea_sum))]
    windowed_chorea = np.array(windowed_chorea)
    windowed_chorea = np.expand_dims(windowed_chorea, axis=-1)
    windowed_labels = mode(windowed_labels, axis=1)[0]
    # Concatenate the data from different subjects
    windowed_data_all = np.append(windowed_data_all, windowed_data, axis=0)
    windowed_labels_all = np.append(windowed_labels_all, windowed_labels, axis=0)
    windowed_chorea_all = np.append(windowed_chorea_all, windowed_chorea, axis=0)

    return windowed_data_all, windowed_labels_all, windowed_chorea_all, inclusion_idx, NumWin

def tensor_data_loader(windowed_data_all, windowed_labels_all, device, batch_size):
    """ Converting the numpy data into torch tensor format
      :param windowed_data_all: NumPy array of shape (n_windows, 1, 3, window_size)
      :param windowed_labels_all: NumPy array of shape (n_windows, 1)
      :param batch_size: Integer
      :param device: Cuda/CPU

      :return: Tensor Dataloader
      """
    tensor_x = torch.Tensor(windowed_data_all).float().to(device)  # transform np to torch tensor
    tensor_y = torch.Tensor(windowed_labels_all).float().to(device)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)  # create your dataloader
    return tensor_dataloader
