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


def test_w_shuffle_split(X, y, ppi):
    
    ppi_kernel_ = ppi_kernel(ppi=ppi, gamma_n=4, gamma_alpha=5)
    svr_en_ppi = ppi_kernel(ppi=ppi, gamma_n=4, gamma_alpha=10, n_estimators=20, alpha_factor=3, n_proteins_ensemble=30)
    svr_en_ppi_random = ppi_kernel(ppi=ppi, gamma_n=4, gamma_alpha=5, n_estimators=3, alpha_factor=3)

    test_size = np.linspace(0.55, 0.15, 10)
    n_splits = 10

    r2_ppi = np.zeros((n_splits, len(test_size)))
    r2_ppi_en = np.zeros((n_splits, len(test_size)))
    r2_ppi_random = np.zeros((n_splits, len(test_size)))
    r2_rfr = np.zeros((n_splits, len(test_size)))
    r2_linear = np.zeros((n_splits, len(test_size)))

    for t_index, t_size in enumerate(test_size):

        ss = ShuffleSplit(n_splits=n_splits, test_size=t_size, random_state=0)

        split = 0

        for train_index, test_index in ss.split(X):

            X_train, X_test = X[train_index, :], X[test_index, :]

            y_train, y_test = y[train_index], y[test_index]

            '''
            PPI SVR using whole interaction matrix
            '''

            train_kernel, D = ppi_kernel_.compute_interaction_kernel(data=X_train, norm_kernel=True)

            svr_ppi = SVR(kernel='precomputed', C=20)

            svr_ppi.fit(train_kernel, y_train)

            svm_predict_data = [X_test, X_train]

            predict_kernel, _ = ppi_kernel_.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                                   norm_kernel=True, D=D)

            y_pred_ppi = svr_ppi.predict(predict_kernel)

            ''' PPI_SVR using Ensembles
            '''

            svr_en_ppi.fit_ensemble_svm(X_train, y_train)

            y_pred_ppi_en = svr_en_ppi.ensemble_predict(X_test)

            ''' PPI_SVR using random sampling
            '''

            svr_en_ppi_random.fit_ensemble_svm(X_train, y_train, random=True)

            y_pred_ppi_random = svr_en_ppi_random.ensemble_predict(X_test)

            '''
            Linear SVR
            '''

            linear_svr = SVR(kernel='linear')

            linear_svr.fit(X_train, y_train)

            y_pred_linear_kernel = linear_svr.predict(X_test)

            '''Random Forests
            '''

            rfr = RandomForestRegressor(n_estimators=100)

            rfr.fit(X_train, y_train)

            y_pred_rfr = rfr.predict(X_test)

            ''' Compute Score
            '''

            r2_ppi_ = r2_score(y_test, y_pred_ppi)
            r2_ppi_en_ = r2_score(y_test, y_pred_ppi_en)
            r2_ppi_random_ = r2_score(y_test, y_pred_ppi_random)
            r2_linear_ = r2_score(y_test, y_pred_linear_kernel)
            r2_rfr_ = r2_score(y_test, y_pred_rfr)

            r2_ppi[split, t_index] = r2_ppi_
            r2_ppi_en[split, t_index] = r2_ppi_en_
            r2_ppi_random[split, t_index] = r2_ppi_random_
            r2_linear[split, t_index] = r2_linear_
            r2_rfr[split, t_index] = r2_rfr_

            print('R2_scores: ppi:{}, ppi_en:{}, '
                  'ppi_en_rd:{}, l_svr:{}, rfr:{}'.format(r2_ppi_, r2_ppi_en_, r2_ppi_random_,
                                                          r2_linear_, r2_rfr_))

            split = split + 1

    np.save('r2_ppi', r2_ppi)
    np.save('r2_ppi_en', r2_ppi_en)
    np.save('r2_ppi_en_random', r2_ppi_random)
    np.save('r2_linear_svm', r2_linear)
    np.save('r2_rfr', r2_rfr)

