import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from Pacific_analysis.PPI import ppi_kernel
from Pacific_analysis.utils import nrmse

'''
Performance test of PPI_kernel and its variants against Random Forests and Linear SVM,
using Shuffle Split and R2_score
'''


def plot_y_vs_pred(X, y, ppi):

    test_size = 0.3
    gamma_n_value = 3
    gamma_alpha_value = 2
    C = 4
    epsilon = 0.3

    n_splits = 3

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=2)

    for train_index, test_index in ss.split(X):

        X_train, X_test = X[train_index, :], X[test_index, :]

        y_train, y_test = y[train_index], y[test_index]

        rfr = RandomForestRegressor(n_estimators=1200, max_depth=15,
                                    min_samples_leaf=2, min_samples_split=2)
        rfr.fit(X_train, y_train)
        rfr_pred = rfr.predict(X_test)
        rfr_pred_ = (rfr_pred - np.mean(rfr_pred)) / (np.std(rfr_pred))
        y_test_ = (y_test-np.mean(y_test))/np.std(y_test)
        rf_r2 = r2_score(rfr_pred, y_test)
        rf_r2_ = r2_score(rfr_pred_, y_test_)
        nrmse_rf = nrmse(y_test, rfr_pred, norm_factor='range')

        '''
        Ensemble PPI
        '''

        n_estimators = 60
        alpha_factor = 4
        n_proteins_ensemble = int(0.2*(np.shape(X)[1]))

        ppi_kernel_en = \
            ppi_kernel(ppi=ppi, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value,
                       n_estimators=n_estimators, alpha_factor=alpha_factor,
                       n_proteins_ensemble=n_proteins_ensemble)

        ppi_kernel_en.fit_ensemble_svm(X_train, y_train)
        # y_train_pred_en = ppi_kernel_en.ensemble_predict(X_train)
        y_pred_ppi_en = ppi_kernel_en.ensemble_predict(X_test)
        nrmse_ppi_en = \
            nrmse(y_test, y_pred_ppi_en, norm_factor='range')
        r2_ppi_en = \
            r2_score(y_test, y_pred_ppi_en)

        '''
        Whole network PPI
        '''
        ppi_kernel_ = \
            ppi_kernel(ppi=ppi, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value)

        train_kernel, D = \
            ppi_kernel_.compute_interaction_kernel(data=X_train, norm_kernel=True)

        svm_predict_data = [X_test, X_train]

        predict_kernel, _ = \
            ppi_kernel_.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                   norm_kernel=True, D=D)

        svr_ppi = SVR(kernel='precomputed', C=C, epsilon=epsilon)

        svr_ppi.fit(train_kernel, y_train)
        y_pred_ppi = svr_ppi.predict(predict_kernel)

        r2_score_ppi = r2_score(y_pred_ppi, y_test)
        nrmse_ppi = nrmse(y_pred_ppi, y_test, norm_factor='range')

        '''
        Plotting results, Predicted vs Actual Values
        '''
#       plt.close()
        fig = plt.figure()
        fig.suptitle('Using Best NRMSE HyperParams for 150 Features Regression')
        plt.subplot(3, 1, 1)
        plt.plot(y_test, label='true')
        plt.plot(y_pred_ppi_en, label='predicted_ppi_en r2:{}, '
                                      'nrmse:{}'.format(r2_ppi_en, nrmse_ppi_en))
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(y_test, label='true')
        plt.plot(y_pred_ppi, label='predicted_ppi r2:{}, '
                                      'nrmse:{}'.format(r2_score_ppi, nrmse_ppi))
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(y_test, label='true')
        plt.plot(rfr_pred, label='predicted_rf r2:{}, '
                                      'nrmse:{}'.format(rf_r2, nrmse_rf))
        plt.legend()
        plt.show()
