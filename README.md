# Nested Cross-Validation for scikit-learn using MPI

This package provides nested cross-validation similar to
scikit-learn's [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html)
but uses the Message Passing Interface (MPI) for parallel computing.

## Requirements

* [scikit-learn](http://scikit-learn.org) 0.16.0 or later
* [mpi4py](http://mpi4py.scipy.org)
* [pandas](http://pandas.pydata.org)

## Example

```python
from mpi4py import MPI
import numpy
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from grid_search import NestedGridSearchCV

data = load_boston()
X = data['data']
y = data['target']

estimator = SVR(max_iter=1000, tol=1e-5)

param_grid = {'C': 2. ** numpy.arange(-5, 15, 2),
              'gamma': 2. ** numpy.arange(3, -15, -2),
              'kernel': ['poly', 'rbf']}

nested_cv = NestedGridSearchCV(estimator, param_grid, 'mean_absolute_error',
                               cv=5, inner_cv=3)
nested_cv.fit(X, y)

if MPI.COMM_WORLD.Get_rank() == 0:
    for i, scores in enumerate(nested_cv.grid_scores_):
        scores.to_csv('grid-scores-%d.csv' % (i + 1), index=False)

    print(nested_cv.best_params_)
```

The result should look like this:

core (Validation) |	C | gamma | kernel | score (Test)
----------------- | - | ----- | ------ | ------------
fold | | | |
1 |	-7.252490 |	0.5 |	0.000122 |	rbf |	-4.178257
2 |	-5.662221 |	128.0 |	0.000122 |	rbf |	-5.445915
3 |	-5.582780 |	32.0 |	0.000122 |	rbf |	-7.066123
4 |	-6.306561 |	0.5 |	0.000122 |	rbf |	-6.059503
5 |	-6.174779 |	128.0 |	0.000122 |	rbf |	-6.606218
