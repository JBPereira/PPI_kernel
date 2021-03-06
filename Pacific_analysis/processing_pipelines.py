from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import DistanceMetric
import numpy as np
from Pacific_analysis.utils import create_artificial_samples


class NN_imputer(BaseEstimator, TransformerMixin):
    def __init__(self, metric='euclidean', n_neighbors=3):
        self.metric = metric
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dist = DistanceMetric.get_metric(self.metric)  # used to calculate NNs

        sum_columns = np.sum(X, axis=1)

        incomplete_rows_index = np.argwhere(
            np.isnan(sum_columns)).flatten()  # select rows where there is any missing value
        donors_rows_index = np.argwhere(np.isnan(sum_columns) == False).flatten()  # select complete rows

        donors, incomplete_rows = X[donors_rows_index, :], X[incomplete_rows_index, :]

        donors = np.ma.array(donors, mask=False)

        for index, incomplete in enumerate(incomplete_rows):
            missing_values_index = np.argwhere(np.isnan(incomplete)).flatten()  # the missing variables for this row

            donors_mask = np.ma.array(donors,
                                      mask=False)  # create mask of donor matrix to exclude the missing values when computing nearest neighbors

            donors_mask.mask[:, missing_values_index] = True

            sorted_neighbors = np.argsort(dist.pairwise(donors_mask, [
                incomplete]))  # compute nearest neighbor using available variables from incomplete row

            nn_relevant_values = donors[:self.n_neighbors,
                                 missing_values_index]  # the nns' values for the incomplete row missing variables

            X[incomplete_rows_index[index], missing_values_index] = np.mean(nn_relevant_values, axis=0)

        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, drop_na_rows=False):

        '''
        :param columns_to_drop: List of columns to drop
        :param drop_na_rows: string: 'all' drops rows that have NaNs only, 'any' drops rows that have any NaN
        '''

        self.columns_to_drop = columns_to_drop
        self.drop_na_rows = drop_na_rows

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if self.columns_to_drop is not None:
            X.drop(columns=self.columns_to_drop, inplace=True)

        if self.drop_na_rows is not False:
            X.dropna(how=self.drop_na_rows, inplace=True)  # drop rows that do not have any value

        return X.values


class DataNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='z-score'):

        self.method = method

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if self.method == 'z-score':
            data_mean = np.nanmean(X, axis=0)
            data_std = np.nanstd(X, axis=0)

            norm_data = (X - data_mean) / data_std

            return norm_data


        elif self.method == 'minmax':

            data_min = np.nanmax(X, axis=0)
            data_max = np.nanmin(X, axis=0)

            norm_data = (X - data_min) / (data_max - data_min)

            return norm_data


class SubstituteBelowDetection(BaseEstimator, TransformerMixin):

    def __init__(self, std_percentage=0.01, repeats=5, lower_percentile=10):

        self.std_percentage = std_percentage
        self.repeats = repeats
        self.lower_percentile = lower_percentile

    def fit(self, X, y=None):

        return self

    def transform(self, X, y):

        above_detection = y > 0

        X_above_detection = X[above_detection, :]
        y_above_detection = y[above_detection]

        low_mid_percentile = np.percentile(y_above_detection, self.lower_percentile)

        low_mid_X = X_above_detection[y_above_detection < low_mid_percentile, :]

        low_mid_y = y_above_detection[y_above_detection < low_mid_percentile]

        X_noise, y_noise = create_artificial_samples(low_mid_X, low_mid_y, repeats=self.repeats,
                                                     std_percentage=self.std_percentage)

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

        y[y == 0] = random_search.predict(X[y == 0])

        return X


def pipeline_w_norm(X, columns_to_drop):

    pipeline = Pipeline([
        ('selector', DataFrameSelector(columns_to_drop)),
        ('normalizer', DataNormalizer(method='z-score')),
        ('NNimputer', NN_imputer(n_neighbors=3))
    ])

    return pipeline.fit_transform(X)


def normalize_matrix(X, method='z-score'):

    return DataNormalizer(method).fit_transform(X)


def pipeline_wout_norm_or_imputation(X, columns_to_drop, drop_na_rows=False):

    pipeline = Pipeline([
    ('selector', DataFrameSelector(columns_to_drop=columns_to_drop, drop_na_rows=drop_na_rows)),
    ])

    return pipeline.fit_transform(X)


def pipeline_wout_imputation(X, columns_to_drop=False, drop_na_rows=False):

    pipeline = Pipeline([
        ('selector', DataFrameSelector(columns_to_drop=columns_to_drop, drop_na_rows=drop_na_rows)),
        ('normalizer', DataNormalizer(method='z-score')),
    ])

    return pipeline.fit_transform(X)