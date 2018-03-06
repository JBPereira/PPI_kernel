import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from Pacific_analysis.PPI import ppi_kernel

'''
Performance test of PPI_kernel and its variants against Random Forests and Linear SVM,
using Shuffle Split and R2_score
'''


def test_ppi_hyper(X, y, ppi):

    test_size = np.linspace(0.55, 0.15, 5)
    gamma_n = np.arange(2, 5)
    gamma_alpha = np.arange(1, 10)
    C_array = np.geomspace(0.1, 100, 10)

    n_splits = 5

    r2_ppi = np.zeros((n_splits, len(test_size), len(gamma_n), len(gamma_alpha), len(C_array)))

    for t_index, t_size in enumerate(test_size):

        ss = ShuffleSplit(n_splits=n_splits, test_size=t_size, random_state=0)

        split = 0

        for train_index, test_index in ss.split(X):

            X_train, X_test = X[train_index, :], X[test_index, :]

            y_train, y_test = y[train_index], y[test_index]

            for gamma_n_index, gamma_n_value in enumerate(gamma_n):

                for gamma_alpha_index, gamma_alpha_value in enumerate(gamma_alpha):

                    for C_index, C in enumerate(C_array):

                        ppi_kernel_ = \
                            ppi_kernel(ppi=ppi, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value)

                        train_kernel, D = \
                            ppi_kernel_.compute_interaction_kernel(data=X_train, norm_kernel=True)

                        svr_ppi = SVR(kernel='precomputed', C=C)

                        svr_ppi.fit(train_kernel, y_train)

                        svm_predict_data = [X_test, X_train]

                        predict_kernel, _ = \
                            ppi_kernel_.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                                   norm_kernel=True, D=D)

                        y_pred_ppi = svr_ppi.predict(predict_kernel)

                        r2_ppi[split, t_index, gamma_n_index, gamma_alpha_index, C_index] = \
                            r2_score(y_pred_ppi, y_test)
            split = split + 1

    average_r2 = np.mean(np.mean(r2_ppi, axis=0), axis=0)

    return average_r2, gamma_n, gamma_alpha, C_array