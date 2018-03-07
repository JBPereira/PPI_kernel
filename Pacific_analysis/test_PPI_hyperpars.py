import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from Pacific_analysis.PPI import ppi_kernel

'''
Performance test of PPI_kernel and its variants against Random Forests and Linear SVM,
using Shuffle Split and R2_score
'''


def test_ppi_hyper(X, y, ppi):

    test_size = 0.3
    gamma_n = np.arange(2, 5)
    gamma_alpha = np.arange(1, 5)
    C_array = np.linspace(5, 30, 4)
    epsilon_array = np.linspace(0.01, 0.3, 5)

    n_splits = 3

    r2_ppi = np.zeros((n_splits, len(gamma_n), len(gamma_alpha),
                       len(C_array), len(epsilon_array)))
    r2_ppi_en = np.zeros((n_splits, len(gamma_n), len(gamma_alpha),
                         len(C_array), len(epsilon_array)))

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=2)

    split = 0

    for train_index, test_index in ss.split(X):

        X_train, X_test = X[train_index, :], X[test_index, :]

        y_train, y_test = y[train_index], y[test_index]

        for gamma_n_index, gamma_n_value in enumerate(gamma_n):

            for gamma_alpha_index, gamma_alpha_value in enumerate(gamma_alpha):

                for C_index, C in enumerate(C_array):

                    for eps_index, eps in enumerate(epsilon_array):

                        ppi_kernel_ = \
                            ppi_kernel(ppi=ppi, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value)

                        ppi_kernel_en = \
                            ppi_kernel(ppi=ppi, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value,
                                       n_estimators=30, alpha_factor=3, n_proteins_ensemble=30)

                        train_kernel, D = \
                            ppi_kernel_.compute_interaction_kernel(data=X_train, norm_kernel=True)

                        svr_ppi = SVR(kernel='precomputed', C=C, epsilon=eps)

                        svr_ppi.fit(train_kernel, y_train)

                        ppi_kernel_en.fit_ensemble_svm(X_train, y_train)

                        svm_predict_data = [X_test, X_train]

                        predict_kernel, _ = \
                            ppi_kernel_.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                                   norm_kernel=True, D=D)

                        y_pred_ppi = svr_ppi.predict(predict_kernel)
                        y_pred_ppi_en = ppi_kernel_en.ensemble_predict(X_test)

                        r2_score_ = r2_score(y_pred_ppi, y_test)
                        r2_score_en_ = r2_score(y_pred_ppi_en, y_test)
                        rfr = RandomForestRegressor(n_estimators=800, max_depth=10)
                        rfr.fit(X_train, y_train)
                        rfr_pred = rfr.predict(X_test)
                        rf_r2 = r2_score(rfr_pred, y_test)

                        '''
                        Plotting results, Predicted vs Actual Values
                        '''
                        plt.figure(figsize=(15, 15))
                        plt.subplot(2, 1, 1)
                        y_train_pred = svr_ppi.predict(train_kernel)
                        y_train_pred_en = ppi_kernel_en.ensemble_predict(X_train)
                        r2_score_train = r2_score(y_train, y_train_pred)
                        r2_score_train_en = r2_score(y_train, y_train_pred_en)
                        plt.plot(y_train, 'b', label='y_train')
                        plt.plot(y_train_pred, 'r', label='ppi_train_pred r2_score:{}'.format(r2_score_train))
                        plt.plot(y_train_pred_en, 'black',
                                 label='ppi_train_pred_en r2_score:{}'.format(r2_score_train_en))
                        plt.legend()
                        plt.subplot(2, 1, 2)
                        plt.plot(y_test, 'b', label='y_test')
                        plt.plot(y_pred_ppi, 'r', label='ppi_pred r2_score:{}'.format(r2_score_))
                        plt.plot(rfr_pred, 'g', label='rfr_pred r2_score:{}'.format(rf_r2))
                        plt.plot(y_pred_ppi_en, 'black', label='ppi_pred_en r2_score:{}'.format(r2_score_en_))
                        plt.legend()
                        plt.show()

                        r2_ppi[split, gamma_n_index, gamma_alpha_index, C_index, eps_index] = \
                            r2_score_
                        r2_ppi_en[split, gamma_n_index, gamma_alpha_index, C_index, eps_index] = \
                            r2_score_en_
        split += 1

    average_r2 = np.mean(r2_ppi, axis=0)
    average_r2_en = np.mean(r2_ppi_en, axis=0)

    return average_r2, average_r2_en, gamma_n, gamma_alpha, C_array, epsilon_array
