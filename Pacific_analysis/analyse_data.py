import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFECV, f_regression
from sklearn.ensemble import ExtraTreesRegressor

from Pacific_analysis.processing_pipelines import pipeline_wout_norm_or_imputation, \
    normalize_matrix


'''
Read Data and Pre process it
'''

columns_to_drop = ['Unnamed: 3', 'Unnamed: 4']

plaque_markers_data = pd.read_excel('data.xlsx')

processed_X = pipeline_wout_norm_or_imputation(plaque_markers_data, columns_to_drop=columns_to_drop, drop_na_rows='any')

y = processed_X[:, 2]

normal_values = np.argwhere(y < 300).flatten()

processed_X = processed_X[normal_values, :]

# processed_X[:, 4:] = 2 ** processed_X[:, 4:]

processed_norm_X = normalize_matrix(processed_X, method='z-score')

columns = np.ma.array(plaque_markers_data.columns, mask=False)
columns.mask[3:4] = True

processed_X_df = pd.DataFrame(processed_X, columns=columns)
processed_X_norm_df = pd.DataFrame(processed_norm_X, columns=columns)

'''
Read Protein interactions table and create the matrix
'''

protein_interactions_table = pd.read_csv('protein_interactions.csv', sep='\t')

protein_interactions_table['exp_text_avg'] = (protein_interactions_table['experimentally_determined_interaction'] +
                                              protein_interactions_table['automated_textmining']) / 2

proteins_in_data = plaque_markers_data.columns[4:]

unique_proteins_1 = protein_interactions_table.iloc[:, 0].unique()
unique_proteins_2 = protein_interactions_table.iloc[:, 1].unique()

unique_proteins = np.unique(np.r_[unique_proteins_1, unique_proteins_2])

common_proteins = np.intersect1d(unique_proteins, proteins_in_data)

mask = np.in1d(unique_proteins, proteins_in_data, invert=True)

mask_first_column = np.in1d(protein_interactions_table.iloc[:, 0], proteins_in_data)
mask_second_column = np.in1d(protein_interactions_table.iloc[:, 1], proteins_in_data)

mask_proteins = np.logical_and(mask_first_column, mask_second_column)

reduced_protein_table = protein_interactions_table.iloc[mask_proteins, :]

proteins_in_interaction_only = unique_proteins[mask]

interaction_matrix = pd.DataFrame(0, index=plaque_markers_data.columns[4:], columns=plaque_markers_data.columns[4:])

for row in reduced_protein_table.loc[:, ['#node1', 'node2', 'exp_text_avg']].values:
    protein_1 = row[0]
    protein_2 = row[1]
    interaction_matrix.loc[protein_1, protein_2] = row[2]
    interaction_matrix.loc[protein_2, protein_1] = row[2]

for i in range(interaction_matrix.shape[0]):
    interaction_matrix.iloc[i, i] = 1

interaction_matrix.fillna(0, inplace=True)

total_interation_value = np.sum(interaction_matrix, axis=1)
n_non_interacting_proteins = len(np.argwhere(total_interation_value == 0)) / len(total_interation_value)

'''
Test PPI_kernel method against other Regressors
'''

X = processed_X_norm_df.iloc[:, 4:].values

# random_cols = np.random.choice(np.arange(0, X.shape[1]), 80)

# X = X[:, random_cols]

y = processed_X_norm_df.iloc[:, 1].values

'''
Protein Pre-selection using linear SVC. Comment to use whole network
'''

### RFE with RF

# n_estimators = 100
# cv = 3

# estimator = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
# selector = RFECV(estimator, step=1, cv=3)
# selector = selector.fit(X, y)
# selected_features = selector.get_support()
# np.save('selected_features_RF_{}_est_{}_cv'.format(n_estimators, cv), selected_features)


### Select only features whose p-value is <0.05

# selector = SelectKBest(f_regression, k='all').fit(X, y)
# p_values = selector.pvalues_
# selected_features = np.argwhere(p_values < 0.05).flatten()
# np.save('selected_features_skb_f_reg_p<0.05', selected_features)
#
# print('\n\n\n SELECTED_FEATURES '
#       'P-VALUE: \n\n')
# for i in selected_features:
#     print('Feature: {} P-Value: {}'.format(processed_X_norm_df.iloc[:, 4:].columns[i],
#                                            p_values[i]))

### Select k features

k = 100
selector = SelectKBest(f_regression, k=k).fit(X, y)
selected_features = selector.get_support()
# np.save('selected_features_skb_f_reg_{}'.format(k), selected_features)
#
# X = X[:, selected_features]
#
# ppi = interaction_matrix.iloc[selected_features, selected_features]

print('\n\n\n SHAPE X: {} \n\n\n'.format(np.shape(X)))

'''
Quick-Test with one split, comparing kernel_PPI, ensemble_PPI, random_ensemble_PPI, RF and linear SVM
'''

from Pacific_analysis.quick_test import quick_test

train_size = 0.6

# quick_test(X, y, ppi, train_size=train_size)

'''
Plot PPI vs y_true
'''

from Pacific_analysis.plot_y_vs_pred import plot_y_vs_pred

# plot_y_vs_pred(X, y, ppi)

'''
Compare PPI performance on normal dataset vs dataset with zeros substituted using a Regressor on
low non-zero y values with added noise
'''

from Pacific_analysis.compare_zero_substitution_performance import compare_zero_substitution_performance

compare_zero_substitution_performance(X, y, y)

'''
Test PPI_kernel Hyperparameters
'''

# from Pacific_analysis.test_PPI_hyperpars import test_ppi_hyper
# import json
#
# ppi_variables, ppi_en_variables, rf_variables = test_ppi_hyper(X, y, interaction_matrix, feat_elim=100)
#
# average_r2_ppi, average_nrmse_ppi, gamma_n, gamma_alpha, C_array, eps_array = ppi_variables
# average_r2_ppi_en, average_nrmse_ppi_en, n_estimators, alpha_factor, n_proteins_ensemble = ppi_en_variables
# average_r2_rf, average_nrmse_rf = rf_variables
#
# selected_feature_names = processed_X_norm_df.iloc[:, 4:].columns[selected_features].tolist()
#
# to_save = [{'feat_elim (inside split)': str(selector.get_params()),
#             'average_r2_ppi': average_r2_ppi.tolist(),
#             'average_nrsme_ppi': average_nrmse_ppi.tolist(),
#             'gamma_n': gamma_n.tolist(),
#             'gamma_alpha': gamma_alpha.tolist(),
#             'C_array': C_array.tolist(),
#             'eps_array': eps_array.tolist(),
#             # 'selected_features (Inside split)': selected_feature_names,
#             # 'selected_features_len': len(selected_feature_names)
#             },
#            {'feat_elim': str(selector.get_params()),
#             'n_estimators_nrmse_selected': n_estimators,
#             'alpha_factor': alpha_factor,
#             'n_proteins_ensemble': n_proteins_ensemble,
#             'average_r2_ppi_en': average_r2_ppi_en.tolist(),
#             'average_nrmse_ppi_en': average_nrmse_ppi_en.tolist()},
#            {'feat_elim': str(selector.get_params()),
#             'average_r2_rf': average_r2_rf.tolist(),
#             'average_nrmse_rf': average_nrmse_rf.tolist()}]
#
# import os.path
#
# if os.path.isfile('feat_elim_scores'):
#     with open('feat_elim_scores', 'r') as f:
#         file = json.load(f)
#     file.append(to_save)
#     with open('feat_elim_scores', 'w') as f:
#         json.dump(file, f)
# else:
#     with open('feat_elim_scores', 'w') as f:
#         json.dump(to_save, f)

'''
Full Test with shuffle_split comparing the same as above
'''

from Pacific_analysis.test_w_shuffle_split import test_w_shuffle_split

# test_w_shuffle_split(X, y, ppi)


'''
Manifold Visualization of data + visualization using TSNE with ppi_kernel
'''

random_cols = np.random.choice(np.arange(0, X.shape[1]), 80)

X = X[:, random_cols]

ppi = interaction_matrix.iloc[random_cols, random_cols]

from Pacific_analysis.manifold_representation import manifold_rep

# manifold_rep(X, y, ppi)
