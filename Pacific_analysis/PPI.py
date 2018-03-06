import numpy as np
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from functools import reduce
import time

'''
PPI kernel method class. Can be operated using random or biased ensembles of regressors,
 or a single regressor using the whole Protein-Protein Interaction Matrix
'''

class ppi_kernel():
    def __init__(self, ppi, gamma_n, gamma_diag=False, gamma_alpha=1,
                 diagonal=False, plain_array=True, n_proteins_ensemble=False,
                 n_estimators=10, alpha_factor=10):

        '''
        Class for PPI SVM regression. Can be used as single PPI SVM or ensemble of PPI SVM regressors

        :param ppi: Protein Protein interaction matrix
        :param gamma_n: maximum length of walks allowed in the PPI matrix to compute the graph kernel
        :param gamma_alpha: how much to penalize long walks in the graph
        :param n_proteins_ensemble: how much proteins to use in each regressor if using ensemble. int or 0<float<1
        :param n_estimators: how many regressors to use if using ensemble
        :param alpha_factor: Probability ratio desired between higher connected and lower connected nodes,
        when sampling biased for ensemble
        '''

        self.ppi = ppi
        self.n_proteins = len(ppi)

        self.gamma = self.compute_exp_decay_gamma(gamma_n, alpha=gamma_alpha, diagonal=diagonal,
                                                  plain_array=plain_array)

        self.T_matrix = self.compute_transition_matrix(alpha_factor)
        self.n_estimators = n_estimators

        if not n_proteins_ensemble:

            self.n_proteins_ensemble = int(np.sqrt(self.n_proteins))

        elif n_proteins_ensemble > 1:

            self.n_proteins_ensemble = n_proteins_ensemble

        elif n_proteins_ensemble > 0:

            self.n_proteins_ensemble = int(n_proteins_ensemble * self.n_proteins)

    @staticmethod
    def mad(data):

        flat_array = data.flatten()

        n_elements = len(flat_array)

        diff_array = np.subtract.outer(flat_array, flat_array)

        return np.sum(np.abs(diff_array)) / (np.multiply(n_elements, (n_elements - 1)))

    @staticmethod
    def mp(matrix, power_number):

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

    def compute_interaction_kernel(self, data, D=None, ppi=[], predicting=False, finishing_time=False, norm_kernel=False):

        '''
        Compute the interaction kernel. Can be used both for training and for predicting:
        Training: Pass in training data as a 2-D array
        Prediction: Pass in a list with the test samples in first entry, training samples on the second.
        WARNING: Make sure to pass D, the normalization factor obtained with training, when computing prediction kernel

        '''

        if len(ppi) == 0:
            ppi = self.ppi

        if predicting:

            n_rows = np.shape(data[0])[0]
            n_cols = np.shape(data[1])[0]

            L_array = np.array([np.outer(data[0][i, :], data[0][i, :]) for i in
                                range(n_rows)])  # Edge weight matrix for test samples
            L_prime_array = np.array([np.outer(data[1][i, :], data[1][i, :]) for i in
                                      range(n_cols)])  # Edge weight matrix for train samples

        else:

            L_array = np.array([np.outer(data[i, :], data[i, :]) for i in
                                range(len(data))])  # Edge weight matrix for train samples
            L_prime_array = L_array

        kernel_matrix = self.compute_kernel(L_array, L_prime_array, ppi=ppi, predicting=predicting,
                                            finishing_time=finishing_time)

        if norm_kernel:

            if not predicting:

                D = np.diag(1. / np.sqrt(np.diag(kernel_matrix)))
                D_prime = D

            else:

                kernel_test = self.compute_kernel(L_array, L_array, ppi, predicting=predicting,
                                                  finishing_time=finishing_time)
                D_prime = np.diag(1. / np.sqrt(np.diag(kernel_test)))

            K_norm = reduce(np.dot, (D_prime, kernel_matrix, D))

            return K_norm, D

        else:

            return kernel_matrix

    def compute_kernel(self, L_array, L_prime_array, ppi, predicting, finishing_time=False):

        '''
        Given the array of weighted protein networks (w.p.n) and the array of w.p.n to compute the
        distance with, returns the PPI kernel
        :param L_array: w.p.n.
        :param L_prime_array: w.p.n. to compare L_array with
        :param ppi: Protein-Protein Interaction Matrix
        :param predicting: If using kernel for predicting, pass True
        :param finishing_time: If you want to know he finishing time, pass True
        :return: PPI Kernel
        '''

        n_rows = len(L_array)
        n_cols = len(L_prime_array)

        kernel_matrix = np.zeros((n_rows, n_cols))

        loop_times = []

        for i in range(n_rows):

            if finishing_time:
                start = time.time()

                print('{} % complete'.format(np.float32((i * n_cols)) / np.float32(n_cols * n_rows) * 100))

            for j in range(n_cols):

                if not predicting and j < i:

                    pass

                else:

                    kernel_matrix[i, j] = self.compute_ikernel_entry(L_array[i], L_prime_array[j], ppi.values)

                    if not predicting:

                        if j < n_rows and i != j:  # The matrix is symmetric, no need to compute twice the same value

                            kernel_matrix[j, i] = kernel_matrix[i, j]

            if finishing_time:
                end = time.time()
                loop_times.append(end - start)
                average_cycle = np.mean(loop_times)
                expected_finishing_time = average_cycle * (n_rows - i + 1) / 2
                print('average cycle time:{} \n expected finishing time: {}'.format(average_cycle,
                                                                                    expected_finishing_time))

        return kernel_matrix

    def compute_ikernel_entry(self, L, L_prime, ppi):

        '''
        Computes the PPI kernel entry given by the inner product of the mapped graphs of two measurements
        :param L: Protein Network with edges weighted by the protein concentrations
        :param L_prime: Same as L, for a different measurement
        :param ppi: Protein-Protein Interaction Matrix
        :return: PPI Kernel entry for L and L_prime
        '''

        gamma_n = np.shape(self.gamma)[0]

        weighted_graph_L_array = np.array(
            np.multiply(self.mp(np.multiply(ppi, L), gamma_n), self.gamma[:, np.newaxis, np.newaxis]))

        weighted_graph_L_prime_array = np.array(
            np.multiply(self.mp(np.multiply(ppi, L_prime), gamma_n), self.gamma[:, np.newaxis, np.newaxis]))

        sum_graph_L_array = np.sum(weighted_graph_L_array, axis=0)

        sum_graph_L_prime = np.sum(weighted_graph_L_prime_array, axis=0)

        result = np.sum(np.multiply(sum_graph_L_array, sum_graph_L_prime))

        return result

    @staticmethod
    def compute_exp_decay_gamma(n, alpha=1, diagonal=False, plain_array=True):

        """
        Computes the Gamma matrix
        :param: diagonal: if True, the matrix will only include non-zero elements for walks of the same length (only diagonal is non-zero)
        """

        exp_array = np.exp(-np.arange(1, n + 1) * alpha)

        if plain_array:

            return exp_array

        else:

            if diagonal:

                exp_decay_gamma = np.diag(exp_array)

            else:

                exp_decay_gamma = np.dot(exp_array[np.newaxis, :], exp_array[:, np.newaxis])

            return exp_decay_gamma

    def compute_transition_matrix(self, alpha_factor):

        '''
        Calculates transition matrix whose probability depends both on PPI connection value and
        connectivity similarity between proteins
        :param: delta: controls how much of an impact PPI value and connectivity have on the probabilities.
        the higher, the more likely it is to select groups of proteins that interact, or share similar connectivity
        '''

        c = np.sum(self.ppi.values, axis=0)
        norm_c = c / np.sum(c)
        C = np.tile(norm_c, (len(norm_c), 1))
        c_diff = C - C.T

        delta = self.ppi.values - np.abs(c_diff)

        delta_std = np.std(delta, axis=1)

        alpha = np.power(alpha_factor, np.divide(1, 2 * delta_std))

        T_matrix = np.power(alpha, delta)
        np.fill_diagonal(T_matrix, 0)

        normalization_factor = np.sum(T_matrix, axis=1)

        T_matrix = np.divide(T_matrix, normalization_factor[:, np.newaxis])

        return T_matrix

    def sample_proteins_biased(self, biased_seed):

        '''
        Starting from a random seed, samples the protein-protein interaction matrix, grouping
        connected parts or sparse parts of the network with higher probability.
        :param biased_seed: Seed to start from. If you want to start from random seed do not pass this argument
        :return: sample of proteins
        '''

        if biased_seed:

            first_protein = biased_seed

        else:

            first_protein = np.random.choice(
                self.n_proteins)  ### Consider not doing it randomly but more weight to the connected ones

        protein_list = [first_protein]

        protein_list[0] = first_protein

        for i in range(1, self.n_proteins_ensemble):

            new_protein = np.random.choice(self.n_proteins, 1, p=self.T_matrix[protein_list[i - 1], :])

            while new_protein in protein_list:
                new_protein = np.random.choice(self.n_proteins, 1, p=self.T_matrix[protein_list[i - 1], :])

            protein_list.extend(new_protein)

        return protein_list

    def sample_proteins_random(self):

        '''
        Randomly sample proteins from the network
        :return: random sample of proteins
        '''

        protein_list = np.random.choice(self.n_proteins, self.n_proteins_ensemble, replace=False)

        return protein_list

    def build_kernels(self, X_train, regressors_protein_list, predicting, D_array=None, X_test=False):

        '''
        Builds the kernel SVMs for each of the proteins subgroups sampled
        :param X_train: Training data
        :param regressors_protein_list: List of protein subgroups
        :param predicting: If using this method for predicting, pass True
        :param D_array: If you are predicting, a normalizing matrix for the kernel is necessary. Pass it here
        :param X_test: Test data, if predicting
        :return: Array with the built regressors
        '''

        n_regressors = len(regressors_protein_list)

        if predicting:  # TODO: refactor compute_interaction_kernel input data to prevent RY here

            regressor_kernels = []

            for i in range(n_regressors):
                regressor_kernel, _ = self.compute_interaction_kernel(
                data=[X_test[:, regressors_protein_list[i]], X_train[:, regressors_protein_list[i]]],
                ppi=self.ppi.iloc[regressors_protein_list[i], regressors_protein_list[i]],
                predicting=predicting, norm_kernel=True, D=D_array[i])
                regressor_kernels.append(regressor_kernel)

        else:

            regressor_kernels = [self.compute_interaction_kernel(
                data=X_train[:, regressors_protein_list[i]],
                ppi=self.ppi.iloc[regressors_protein_list[i], regressors_protein_list[i]],
                predicting=predicting, norm_kernel=True) for i in range(n_regressors)]

        return np.array(regressor_kernels)

    def build_regressors(self, X_training, y_training, regressors_protein_list):

        n_regressors = np.shape(regressors_protein_list)[0]

        regressor_kernels = self.build_kernels(X_training, regressors_protein_list, predicting=False)

        kernels = regressor_kernels[:, 0]
        D_array = regressor_kernels[:, 1]

        if len(self.D_array) > 0:
            self.D_array = np.r_[self.D_array, D_array]
        else:
            self.D_array = D_array

        regressors = np.array([[[SVR(kernel='precomputed', C=j, epsilon=k, cache_size=500, shrinking=True).fit(
                                       kernels[i],
                                       y_training)
                                for i in range(n_regressors)]
                                for j in np.geomspace(0.01, 500, 50)]
                               for k in np.linspace(0.01, 0.2, 10)])

        return regressors

    @staticmethod
    def regressors_performance_and_selection(regressors, regressor_predict_kernel, y):

        n_intermediate_regressors_eps, n_intermediate_regressors_C, n_regressors = np.shape(regressors)

        predictions = np.array([[[regressors[k][j][i].predict(regressor_predict_kernel[i])
                                for i in range(n_regressors)]
                                for j in range(n_intermediate_regressors_C)]
                                for k in range(n_intermediate_regressors_eps)])

        r2_score_array = np.array([[[r2_score(predictions[k, j, i, :], y)
                                   for i in range(n_regressors)]
                                   for j in range(n_intermediate_regressors_C)]
                                   for k in range(n_intermediate_regressors_eps)])

        best_predictors_eps = np.argmax(r2_score_array, axis=0)
        grid_indices_eps, grid_indices_reg = np.indices(np.shape(best_predictors_eps))
        r2_eps_dim_collapsed = r2_score_array[best_predictors_eps, grid_indices_eps, grid_indices_reg]
        reg_eps_dim_collapsed = regressors[best_predictors_eps, grid_indices_eps, grid_indices_reg]
        best_predictors_C = np.argmax(r2_eps_dim_collapsed, axis=0)

        regressors = reg_eps_dim_collapsed[best_predictors_C, np.arange(n_regressors)]  # select only best predictors

        r2_score_array = r2_eps_dim_collapsed[best_predictors_C, np.arange(n_regressors)]

        return regressors, r2_score_array

    def seed_and_test_biased_regressors(self, X_training, X_validation, y_training, y_validation,
                                        n_regressors, biased_seed=[]):

        '''
        Samples protein subsets and Evaluates performance of each PPI regressor on the validation set.
        If sampling protein subset using already cherrypicked proteins, pass in the list of proteins numbers
        to seed on the argument biased
        :param X_training: Data to be used to train the regressors
        :param X_validation: Data to be used for Validation
        :param y_training: Target to learn
        :param y_validation: Target to test the regressors
        :param n_regressors: Number of regressors to create/evaluate
        :param biased_seed: If selecting best performance proteins as seeds, pass in list of protein numbers
        :return:
        '''

        if len(biased_seed) == 0:
            biased_seed = [False] * n_regressors

        regressors_protein_list = np.array([self.sample_proteins_biased(biased_seed[i]) for i in range(n_regressors)])

        regressors = self.build_regressors(X_training, y_training, regressors_protein_list)

        regressor_kernels_validation = self.build_kernels(X_train=X_training,
                                                          regressors_protein_list=regressors_protein_list,
                                                          predicting=True, D_array=self.D_array, X_test=X_validation)

        regressors, r2_score_array = self.regressors_performance_and_selection(regressors,
                                                                               regressor_kernels_validation,
                                                                               y_validation)

        return regressors, regressors_protein_list, r2_score_array

    def select_biased_regressors(self, X_training, y_training, X_validation, y_validation):

        explorer_regressors, pioneer_proteins, exploration_scores = self.seed_and_test_biased_regressors(
            X_training=X_training,
            X_validation=X_validation,
            y_training=y_training,
            y_validation=y_validation,
            n_regressors=int(self.n_estimators / 2))

        exploration_mad = self.mad(exploration_scores)

        selected_pioneers = np.array(np.where(exploration_scores >
                                              np.max(exploration_scores - exploration_mad))).flatten()

        selected_regressors = explorer_regressors[selected_pioneers]

        self.D_array = self.D_array[selected_pioneers]

        ##  Use Best protein combinations repeatedly as seeds for remaining regressors

        n_remaining_regressors = self.n_estimators - np.shape(selected_pioneers)[-1]

        n_seed_repeat, n_seed_remainder = np.divmod(n_remaining_regressors, np.shape(selected_pioneers)[-1])

        ordered_scores = np.argsort(exploration_scores[selected_pioneers])

        repeated_seeds_index = np.tile(selected_pioneers[ordered_scores[::-1]],
                                         n_seed_repeat)  # TODO: careful because repeat is not cycling array but repeating each element n times before the next

        remainder_seeds_index = selected_pioneers[ordered_scores[:-n_seed_remainder - 1:-1]]

        selected_protein_seeds = pioneer_proteins[0, np.r_[repeated_seeds_index, remainder_seeds_index]]

        natural_selection_regressors, natural_selection_proteins, \
        natural_selection_scores = self.seed_and_test_biased_regressors(X_training=X_training,
                                                                        X_validation=X_validation,
                                                                        y_training=y_training,
                                                                        y_validation=y_validation,
                                                                        n_regressors=len(selected_protein_seeds),
                                                                        biased_seed=selected_protein_seeds)

        regressors = np.r_[selected_regressors, natural_selection_regressors]
        regressors_proteins = np.r_[pioneer_proteins[selected_pioneers], natural_selection_proteins]
        r2_scores = np.r_[exploration_scores[selected_pioneers], natural_selection_scores]

        cream_of_the_cream_mad = self.mad(r2_scores)  # Perform a final fitness selection of the regressors

        final_selection = np.array(np.where(np.logical_and(r2_scores >
                                            np.max(r2_scores - cream_of_the_cream_mad),
                                                r2_scores > 0))).flatten()

        self.D_array = self.D_array[final_selection]

        self.regressors = regressors[final_selection]

        self.regressors_protein_list = regressors_proteins[final_selection]

        self.X_train = X_training

        self.n_estimators = len(final_selection)

        return r2_scores[final_selection]

    def fit_ensemble_svm(self, X_train, y_train, random=False, holdout=0.2):

        self.D_array = np.array([])

        train_len = int((1 - holdout) * len(X_train))

        X_training = X_train[:train_len, :]
        X_validation = X_train[train_len:, :]
        y_training = y_train[:train_len]
        y_validation = y_train[train_len:]

        if random:

            regressors_protein_list = [self.sample_proteins_random() for i in range(self.n_estimators)]
            regressors = self.build_regressors(X_training=X_training, y_training=y_training,
                                               regressors_protein_list=regressors_protein_list)
            kernel_validation = self.build_kernels(X_train=X_training, regressors_protein_list=regressors_protein_list,
                                                  predicting=True, D_array=self.D_array, X_test=X_validation)

            regressors, r2_score_array = self.regressors_performance_and_selection(regressors=regressors,
                                                                                   regressor_predict_kernel=kernel_validation,
                                                                                   y=y_validation)

            self.regressors = regressors
            self.regressors_protein_list = regressors_protein_list
            self.X_train = X_training

        else:

            r2_score_array = self.select_biased_regressors(X_training=X_training, y_training=y_training,
                                                           X_validation=X_validation, y_validation=y_validation)

        r2_score_array_ = np.power(self.n_estimators, r2_score_array)
        r2_score_array_ /= np.sum(r2_score_array_)

        self.ensemble_weights = r2_score_array_

    def ensemble_predict(self, X_test):

        regressor_kernels = self.build_kernels(X_train=self.X_train,
                                               regressors_protein_list=self.regressors_protein_list,
                                               predicting=True, X_test=X_test, D_array=self.D_array)

        predictions = np.array([self.regressors[i].predict(regressor_kernels[i]) for i in range(self.n_estimators)])

        prediction = np.sum(predictions * np.array(self.ensemble_weights)[:, np.newaxis], axis=0)

        return prediction

