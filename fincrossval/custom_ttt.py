import warnings

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplitLargeTest(_BaseKFold):
    '''
    Time series split train-test split.
    '''

    def __init__(self, n_splits=5, max_train_size=None):
        warnings.simplefilter('always', DeprecationWarning)

        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        assert np.unique(X.index) != len(X.index)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (X.index[indices[test_start - self.max_train_size:test_start]],
                       X.index[indices[test_start:test_start + test_size]])
            else:
                yield (X.index[indices[:test_start]],
                       X.index[indices[test_start:test_start + test_size]])
