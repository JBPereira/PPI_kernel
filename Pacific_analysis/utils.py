import numpy as np
from sklearn.metrics import mean_squared_error


def nrmse(y_pred, y, norm_factor='mean'):

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    if norm_factor == 'range':

        return rmse / (np.max(y) - np.min(y))

    elif norm_factor == 'mean':

        return rmse / np.mean(y)


def mad(data):

    flat_array = data.flatten()

    n_elements = len(flat_array)

    diff_array = np.subtract.outer(flat_array, flat_array)

    return np.sum(np.abs(diff_array)) / (np.multiply(n_elements, (n_elements - 1)))


def m_p(matrix, power_number):

    '''
    Takes a matrix and a power number, and outputs an array of size n with the matrix multiplied by itself
    i times, where n is power_number and i is the index of the array +1
    :param matrix: Matrix to be multiplied by itself
    :param power_number: maximum number of times to multiply matrix by itself
    :return: array with matrix powers
    '''

    if power_number == 1:
        return matrix
    elif power_number == 0:
        return np.abs(matrix)
    else:

        matrix_array = [matrix]

        for i in range(1, power_number):
            matrix_array.append(np.matmul(matrix_array[i - 1], matrix))

    return matrix_array


def select_best(scores, metric='normal', require_positive=False):

    '''
    Select the best scores
    :param scores: array with the scores
    :param metric: If 'mad', select the scores that are above the mean minus the Mean Absolute Deviation
    If normal, select scores that are above the mean minus the standard deviation
    :param require_positive:
    :return: array with indices of best scores
    '''


    if metric == 'mad':

        scores_mad = mad(scores)
        selected_mask = np.ma.array(scores > (np.mean(scores) - scores_mad))

    elif metric == 'normal':

        selected_mask = np.ma.array(scores > np.mean(scores) - np.std(scores))



    if require_positive:


        selected_mask[scores<= 0] = np.ma.masked

    selected_best = np.array(np.where(selected_mask)).flatten()

    return selected_best


def create_artificial_samples(X, y, std_noise=0.01, repeats=5):

    X_std = std_noise * np.std(X, axis=0)

    noise_arrays = np.array([[np.random.normal(0, X_std[i])
                              for i in range(np.shape(X)[1])]
                             for _ in range(repeats*np.shape(X)[0])])

    noise_observations = np.repeat(X, repeats, axis=0) + noise_arrays

    y_noise = np.repeat(y, repeats, axis=0)

    return noise_observations, y_noise

