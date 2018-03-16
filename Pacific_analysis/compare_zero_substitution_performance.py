import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from Pacific_analysis.PPI import ppi_kernel
from Pacific_analysis.utils import nrmse, create_artificial_samples


def compare_zero_substitution_performance(X, y, ppi, n_repeats=5,
                                          std_noise=0.01, lower_percentile=10):

    test_size = 0.3
    gamma_n_value = 3
    gamma_alpha_value = 2
    C = 4
    epsilon = 0.3

    n_splits = 3

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=2)

    below_detection_value = np.min(y)
    above_detection = y > below_detection_value

    X_above_detection = X[above_detection, :]
    y_above_detection = y[above_detection]

    low_mid_percentile = np.percentile(y_above_detection, lower_percentile)

    low_mid_X = X_above_detection[y_above_detection < low_mid_percentile, :]

    low_mid_y = y_above_detection[y_above_detection < low_mid_percentile]

    X_noise, y_noise = create_artificial_samples(low_mid_X, low_mid_y, repeats=n_repeats,
                                                 std_noise=std_noise)

    param_grid = {"n_estimators": [120, 800, 1200],
                  "max_depth": [5, 15, 30, None],
                  "max_features": ["auto", "sqrt", "log2"],
                  "min_samples_leaf": [1, 5, 10],
                  "min_samples_split": [2, 10, 100],
                  "random_state": [512],
                  "bootstrap": [True]}
    trees = ExtraTreesRegressor()
    skf = KFold(n_splits=2)
    random_search = GridSearchCV(trees, param_grid, scoring='r2', cv=skf, n_jobs=-1, verbose=1)

    random_search.fit(X_noise, y_noise)

    y_substituted = y.copy()

    y_substituted[y == below_detection_value] = \
        random_search.predict(X[y == below_detection_value, :])

    for train, test in ss.split(X):
        
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]
        y_subs_train, y_subs_test = y_substituted[train], y_substituted[test]
        
        '''Feature Selection
        '''
        
        selector = SelectKBest(f_regression, k=150).fit(X_train, y_train)
        selected_features = selector.get_support()
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        ppi_selected = ppi.iloc[selected_features, selected_features]
        
        '''Computing PPI kernels for training and predicting'''
        
        ppi_kernel_normal = \
            ppi_kernel(ppi=ppi_selected, gamma_n=gamma_n_value, gamma_alpha=gamma_alpha_value)

        train_kernel_normal, D_normal = \
            ppi_kernel_normal.compute_interaction_kernel(data=X_train, norm_kernel=True)

        svm_predict_data = [X_test, X_train]

        predict_kernel, _ = \
            ppi_kernel_normal.compute_interaction_kernel(data=svm_predict_data, predicting=True,
                                                         norm_kernel=True, D=D_normal)

        ''' Normal y training and predicting
        '''

        svr_ppi_normal = SVR(kernel='precomputed', C=C, epsilon=epsilon)

        svr_ppi_normal.fit(train_kernel_normal, y_train)
        y_pred_ppi = svr_ppi_normal.predict(predict_kernel)

        r2_score_ppi_normal = r2_score(y_pred_ppi, y_test)
        nrmse_ppi_normal = nrmse(y_pred_ppi, y_test, norm_factor='range')

        '''Zero Substituted y Training and Testing'''

        svr_ppi_subs = SVR(kernel='precomputed', C=C, epsilon=epsilon)

        svr_ppi_subs.fit(train_kernel_normal, y_subs_train)
        y_pred_ppi_y_subs = svr_ppi_subs.predict(predict_kernel)

        r2_score_ppi_y_subs = r2_score(y_pred_ppi_y_subs, y_subs_test)
        nrmse_ppi_y_subs = nrmse(y_pred_ppi_y_subs, y_subs_test, norm_factor='range')

        '''
        Plotting results, Predicted vs Actual Values
        '''
        #       plt.close()
        fig = plt.figure()
        fig.suptitle('Comparison y_normal PPI vs y_substituted PPI performance')
        plt.subplot(3, 1, 1)
        plt.plot(y_test, label='true')
        plt.plot(y_pred_ppi, label='predicted_ppi_normal_y r2:{}, '
                                      'nrmse:{}'.format(r2_score_ppi_normal,
                                                        nrmse_ppi_normal))
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(y_test, label='true')
        plt.plot(y_pred_ppi, label='predicted_ppi_y_subs r2:{}, '
                                   'nrmse:{}'.format(r2_score_ppi_y_subs,
                                                     nrmse_ppi_y_subs))
        plt.legend()
        plt.show()
