from Pacific_analysis.PPI import ppi_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score


def quick_test(X, y, interaction_matrix, train_size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    # train_ppi_kernel = PPI_kernel.compute_interaction_kernel(data=X_train, finishing_time=False, norm_kernel=True)

    # svr_ppi = SVR(kernel='precomputed', C=100, epsilon=0.2)
    svr_ppi = ppi_kernel(ppi=interaction_matrix, gamma_n=4, gamma_alpha=3)
    svr_en_ppi = ppi_kernel(ppi=interaction_matrix, gamma_n=4, gamma_alpha=5, n_estimators=30, alpha_factor=4)
    svr_en_ppi_random = ppi_kernel(ppi=interaction_matrix, gamma_n=4, gamma_alpha=20, n_estimators=3, alpha_factor=4)

    svr_en_ppi.fit_ensemble_svm(X_train, y_train)
    svr_en_ppi_random.fit_ensemble_svm(X_train, y_train, random=True)

    whole_train_kernel, D = svr_ppi.compute_interaction_kernel(X_train, norm_kernel=True)

    # svr_ppi.fit(train_ppi_kernel, y_train)

    linear_svr = SVR(kernel='linear', C=100)

    linear_svr.fit(X_train, y_train)

    ppi_svm = SVR(kernel='precomputed', C=50)
    ppi_svm.fit(whole_train_kernel, y_train)
    test_kernel = svr_ppi.compute_interaction_kernel([X_test, X_train],  predicting=True, D=D)

    rfr = RandomForestRegressor(n_estimators=100)

    rfr.fit(X_train, y_train)

    # svm_predict_data = [X_test, X_train]

    # predict_kernel = PPI_kernel.compute_interaction_kernel(data=svm_predict_data, predicting=True, norm_kernel=True)
    # print(predict_kernel)
    y_pred_ppi = ppi_svm.predict(test_kernel)
    y_pred_ppi_ensemble = svr_en_ppi.ensemble_predict(X_test)
    y_pred_ppi_ensemble_random = svr_en_ppi_random.ensemble_predict(X_test)
    y_pred_linear = linear_svr.predict(X_test)
    y_pred_rfr = rfr.predict(X_test)

    for i in range(len(y_test)):
        print('y:{}, rfr:{}, ppi_en:{}, random_ppi:{}, ppi:{}'
              'linear_svm:{}'.format(y_test[i], y_pred_rfr[i], y_pred_ppi_ensemble[i], y_pred_ppi_ensemble_random[i],
                                                                                    y_pred_ppi[i], y_pred_linear[i]))

    print(
        '\n\n R2_SCORE: RFR:{} PPI_SVR:{} PPI_random:{} PPI:{} LINEAR_SVR:{}'.format(r2_score(y_test, y_pred_rfr),
                                                                              r2_score(y_test, y_pred_ppi_ensemble),
                                                                              r2_score(y_test, y_pred_ppi_ensemble_random),
                                                                              r2_score(y_test, y_pred_ppi),
                                                                              r2_score(y_test, y_pred_linear)))
    print(
        '\n\n EXPLAINED_VARIANCE_SCORE: RFR:{} PPI_SVR:{} LINEAR_SVR:{}'.format(
            explained_variance_score(y_test, y_pred_rfr),
            explained_variance_score(y_test, y_pred_ppi_ensemble),
            explained_variance_score(y_test,
                                     y_pred_linear)))