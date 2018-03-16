import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression
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


def test_ppi_hyper(X, y, ppi, feat_elim=1.0):

    test_size = 0.3
    gamma_n = np.arange(3, 6)
    gamma_alpha = np.arange(2, 5)
    C_array = np.linspace(4, 30, 8)
    epsilon_array = np.linspace(0.1, 0.3, 5)

    n_splits = 3

    r2_ppi = np.zeros((n_splits, len(gamma_n), len(gamma_alpha),
                       len(C_array), len(epsilon_array)))
    r2_ppi_en = np.zeros((n_splits, len(gamma_n), len(gamma_alpha)))
    r2_rfr = np.zeros((n_splits, 1))

    nrmse_ppi = np.zeros((n_splits, len(gamma_n), len(gamma_alpha),
                       len(C_array), len(epsilon_array)))
    nrmse_rfr = np.zeros((n_splits, 1))
    nrmse_ppi_en = np.zeros((n_splits, len(gamma_n), len(gamma_alpha)))

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=2)

    split = 0

    for train_index, test_index in ss.split(X):

        X_train, X_test = X[train_index, :], X[test_index, :]

        y_train, y_test = y[train_index], y[test_index]

        if isinstance(feat_elim, (float, int)):

            if feat_elim <= 1:
                k = int(feat_elim * len(X_train))
            else:
                k = feat_elim

            selector = SelectKBest(f_regression, k=k).fit(X_train, y_train)
            selected_features = selector.get_support()

        elif feat_elim == 'p-value':
            selector = SelectKBest(f_regression, k='all').fit(X, y)
            p_values = selector.pvalues_
            selected_features = np.argwhere(p_values < 0.05).flatten()

        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

        ppi_selected = ppi.iloc[selected_features, selected_features]

        rfr = RandomForestRegressor(n_estimators=1200, max_depth=15,
                                    min_samples_leaf=2, min_samples_split=2)
        rfr.fit(X_train, y_train)
        rfr_pred = rfr.predict(X_test)
        rfr_pred_ = (rfr_pred - np.mean(rfr_pred)) / (np.std(rfr_pred))
        y_test_ = (y_test-np.mean(y_test))/np.std(y_test)
        rf_r2 = r2_score(rfr_pred, y_test)
        rf_r2_ = r2_score(rfr_pred_, y_test_)
        r2_rfr[split, 0] = \
            rf_r2
        nrmse_rfr[split, 0] = \
            nrmse(y_test, rfr_pred, norm_factor='range')

        for gamma_n_index, gamma_n_value in enumerate(gamma_n):

            for gamma_alpha_index, gamma_alpha_value in enumerate(gamma_alpha):

                '''
                Ensemble PPI
                '''

                n_estimators = 60
                alpha_factor = 5
                n_proteins_ensemble = int(0.2*(np.shape(X)[1]))

                ppi_kernel_en = \
                    ppi_kernel(ppi=ppi_selected, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value,
                               n_estimators=n_estimators, alpha_factor=alpha_factor,
                               n_proteins_ensemble=n_proteins_ensemble)

                ppi_kernel_en.fit_ensemble_svm(X_train, y_train)
                # y_train_pred_en = ppi_kernel_en.ensemble_predict(X_train)
                y_pred_ppi_en = ppi_kernel_en.ensemble_predict(X_test)
                nrmse_ppi_en[split, gamma_n_index, gamma_alpha_index] = \
                    nrmse(y_test, y_pred_ppi_en, norm_factor='range')
                r2_ppi_en[split, gamma_n_index, gamma_alpha_index] = \
                    r2_score(y_test, y_pred_ppi_en)

                '''
                Whole network PPI
                '''
                ppi_kernel_ = \
                    ppi_kernel(ppi=ppi_selected, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value)

                train_kernel, D = \
                    ppi_kernel_.compute_interaction_kernel(data=X_train, norm_kernel=True)

                svm_predict_data = [X_test, X_train]

                predict_kernel, _ = \
                    ppi_kernel_.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                           norm_kernel=True, D=D)

                for C_index, C in enumerate(C_array):

                    for eps_index, eps in enumerate(epsilon_array):

                        svr_ppi = SVR(kernel='precomputed', C=C, epsilon=eps)

                        svr_ppi.fit(train_kernel, y_train)
                        y_pred_ppi = svr_ppi.predict(predict_kernel)

                        # np.save('y_test_', y_test)
                        # y_pred_ppi_ = (y_pred_ppi-np.mean(y_pred_ppi))/(np.std(y_pred_ppi))
                        # y_pred_ppi_en = (y_pred_ppi_en-np.mean(y_pred_ppi_en))/\
                        #                 (np.std(y_pred_ppi_en))

                        r2_score_ = r2_score(y_pred_ppi, y_test)
                        nrmse_ = nrmse(y_pred_ppi, y_test, norm_factor='range')
                        # r2_score_en_ = r2_score(y_pred_ppi_en, y_test)

                        y_train_pred = svr_ppi.predict(train_kernel)
                        #r2_score_train = r2_score(y_train, y_train_pred)
                        # r2_score_train_en = r2_score(y_train, y_train_pred_en)

                        '''
                        Plotting results, Predicted vs Actual Values
                        '''
    #                    plt.figure(figsize=(15, 15))
                        # plt.subplot(2, 1, 1)
                        #
                        # plt.plot(y_train, 'b', label='y_train')
                        # plt.plot(y_train_pred, 'r', label='ppi_train_pred r2_score:{}'.format(r2_score_train))
                        # plt.plot(y_train_pred_en, 'black',
                        #          label='ppi_train_pred_en r2_score:{}'.format(r2_score_train_en))
                        # plt.legend()
                        # plt.subplot(2, 1, 2)
                       # plt.plot(y_test_, 'b', label='y_test')
                       # plt.plot(y_pred_ppi_, 'r', label='ppi_pred r2_score:{}'.format(r2_score_))
                       # plt.plot(rfr_pred_, 'g', label='rfr_pred r2_score:{}'.format(rf_r2_))
                       #  plt.plot(y_pred_ppi_en, 'black', label='ppi_pred_en r2_score:{}'.format(r2_score_en_))
                       # plt.legend(bbox_to_anchor=(0.85, 0.1))
                       # plt.show()

                        r2_ppi[split, gamma_n_index, gamma_alpha_index, C_index, eps_index] = \
                            r2_score_
                        nrmse_ppi[split, gamma_n_index, gamma_alpha_index, C_index, eps_index] = \
                            nrmse_


                        # r2_ppi_en[split, gamma_n_index, gamma_alpha_index, C_index, eps_index] = \
                        #     r2_score_en_

                        # np.save('r2_ppi_hyper_en', r2_ppi_en)
        split += 1

    average_r2 = np.mean(r2_ppi, axis=0)
    average_r2_ppi_en = np.mean(r2_ppi_en, axis=0)
    average_r2_rf = np.mean(r2_rfr)
    average_nrmse_ppi = np.mean(nrmse_ppi, axis=0)
    average_nrsme_ppi_en = np.mean(nrmse_ppi_en, axis=0)
    average_nrmse_rf = np.mean(nrmse_rfr)

    ppi_variables = [average_r2, average_nrmse_ppi, gamma_n, gamma_alpha, C_array, epsilon_array]
    ppi_en_variables = [average_r2_ppi_en, average_nrsme_ppi_en, n_estimators, alpha_factor, n_proteins_ensemble]
    rf_variables = [average_r2_rf, average_nrmse_rf]

    return ppi_variables, ppi_en_variables, rf_variables
