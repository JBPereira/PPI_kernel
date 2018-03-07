import pandas as pd
import numpy as np

from Pacific_analysis.processing_pipelines import pipeline_wout_norm_or_imputation,\
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

processed_X[:, 4:] = 2 ** processed_X[:, 4:]

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

ppi = interaction_matrix

'''
Quick-Test with one split, comparing kernel_PPI, ensemble_PPI, random_ensemble_PPI, RF and linear SVM
'''

from Pacific_analysis.quick_test import quick_test

train_size = 0.6

# quick_test(X, y, ppi, train_size=train_size)

'''
Test PPI_kernel Hyperparameters
'''

from Pacific_analysis.test_PPI_hyperpars import test_ppi_hyper

average_r2_ppi, average_r2_ppi_en, gamma_n, gamma_alpha, C_array, eps_array = test_ppi_hyper(X, y, ppi)

np.save('r2_ppi_hyper', average_r2_ppi)
np.save('r2_ppi_hyper_gamma_n', gamma_n)
np.save('r2_ppi_hyper_gamma_alpha', gamma_alpha)
np.save('r2_ppi_hyper_C_array', C_array)
np.save('r2_ppi_hyper_eps_array', eps_array)

'''
Full Test with shuffle_split comparing the same as above
'''

from Pacific_analysis.test_w_shuffle_split import test_w_shuffle_split

test_w_shuffle_split(X, y, ppi)



'''
Manifold Visualization of data + visualization using TSNE with ppi_kernel
'''

random_cols = np.random.choice(np.arange(0, X.shape[1]), 80)

X = X[:, random_cols]

ppi = interaction_matrix.iloc[random_cols, random_cols]

from Pacific_analysis.manifold_representation import manifold_rep

# manifold_rep(X, y, ppi)







