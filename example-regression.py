import logging
import numpy

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

from grid_search import NestedGridSearchCV
from sklearn.svm import SVR

data = load_boston()
X = data['data']
y = data['target']
X = StandardScaler().fit_transform(X, y)

estimator = SVR(max_iter=1000, tol=1e-5)

param_grid = {'C': 2. ** numpy.arange(-5, 15, 2),
              'gamma': 2. ** numpy.arange(3, -15, -2),
              'kernel': ['poly', 'rbf']}

kfold_cv = StratifiedKFold(y, n_folds=5)


logging.basicConfig(level=logging.INFO)

nested_cv = NestedGridSearchCV(estimator, param_grid, 'mean_absolute_error', cv=kfold_cv,
                               inner_cv=lambda _x, _y: StratifiedKFold(_y, n_folds=3))
nested_cv.fit(X, y)

from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
    for i, scores in enumerate(nested_cv.grid_scores_):
        scores.to_csv('grid-scores-%d.csv' % (i + 1), index=False)
    print("______________")
    print(nested_cv.best_params_)
