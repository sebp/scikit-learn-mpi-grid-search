import logging

import numpy
from mpi4py import MPI
import pandas
from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import _check_cv, check_scoring, is_classifier, _fit_and_score
from sklearn.grid_search import ParameterGrid, _check_param_grid
from sklearn.utils import check_X_y

__all__ = ['NestedGridSearchCV']

LOG = logging.getLogger(__package__)

MPI_TAG_RESULT = 3

MPI_MSG_TERMINATE = 0
MPI_MSG_CV = 1
MPI_MSG_TEST = 2
MPI_TAG_TRAIN_TEST_DATA = 5

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def _get_best_parameters(fold_results, param_names):
    """Get best setting of parameters from grid search

    Parameters
    ----------
    fold_results : pandas.DataFrame
        Contains performance measures as well as hyper-parameters
        as columns. Must contain a column 'fold'.

    param_names : list
        Names of the hyper-parameters. Each name should be a column
        in ``fold_results``.

    Returns
    -------
    max_performance : pandas.Series
        Maximum performance and its hyper-parameters
    """
    if pandas.isnull(fold_results.loc[:, 'score']).all():
        raise ValueError("Results are all NaN")

    # average across inner folds
    grouped = fold_results.drop('fold', axis=1).groupby(param_names)
    mean_performance = grouped.mean()
    # highest average across performance measures
    max_idx = mean_performance.loc[:, 'score'].idxmax()

    # best parameters
    max_performance = pandas.Series({'score': mean_performance.loc[max_idx, 'score']})
    if len(param_names) == 1:
        key = param_names[0]
        max_performance[key] = max_idx
    else:
        # index has multiple levels
        for i, name in enumerate(mean_performance.index.names):
            max_performance[name] = max_idx[i]

    return max_performance


class MPIBatchWorker(object):
    """Base class to fit and score an estimator"""

    def __init__(self, estimator, scorer, fit_params, verbose=False):
        self.estimator = estimator
        self.scorer = scorer
        self.verbose = verbose
        self.fit_params = fit_params

        # first item denotes ID of fold, and second item encodes
        # a message that tells slaves what to do
        self._task_desc = numpy.empty(2, dtype=int)
        # stores data that the root node broadcasts
        self._data_X = None
        self._data_y = None

    def process_batch(self, work_batch):
        fit_params = self.fit_params if self.fit_params is not None else {}

        LOG.debug("Node %d received %d work items", comm_rank, len(work_batch))

        results = []
        for fold_id, train_index, test_index, parameters in work_batch:
            ret = _fit_and_score(clone(self.estimator),
                                 self._data_X, self._data_y,
                                 self.scorer, train_index, test_index,
                                 self.verbose, parameters, fit_params)

            result = parameters.copy()
            result['score'] = ret[0]
            result['n_samples_test'] = ret[1]
            result['scoring_time'] = ret[2]
            result['fold'] = fold_id
            results.append(result)

        LOG.debug("Node %d is done with fold %d", comm_rank, fold_id)
        return results


class MPISlave(MPIBatchWorker):
    """Receives task from root node and sends results back"""

    def __init__(self, estimator, scorer, fit_params):
        super(MPISlave, self).__init__(estimator, scorer, fit_params)

    def _run_grid_search(self):
        # get data
        self._data_X, self._data_y = comm.bcast(None, root=0)
        # get batch
        work_batch = comm.scatter(None, root=0)

        results = self.process_batch(work_batch)
        # send result
        comm.gather(results, root=0)

    def _run_train_test(self):
        # get data
        self._data_X, self._data_y = comm.bcast(None, root=0)

        work_item = comm.recv(None, source=0, tag=MPI_TAG_TRAIN_TEST_DATA)
        fold_id = work_item[0]
        if fold_id == MPI_MSG_TERMINATE:
            return

        LOG.debug("Node %d is running testing for fold %d", comm_rank, fold_id)

        test_results = self.process_batch([work_item])

        comm.send((fold_id, test_results[0]['score']), dest=0, tag=MPI_TAG_RESULT)

    def run(self):
        """Wait for new data until node receives a message with MPI_MSG_TERMINATE or MPI_MSG_TEST

        In the beginning, the node is waiting for new batches distributed by
        :class:`MPIGridSearchCVMaster._scatter_work`. After the grid search has been completed,
        the node either receives data from :func:`_fit_and_score_with_parameters` to
        evaluate the estimator given the parameters determined during grid-search, or is asked
        to terminate.
        """
        task_desc = self._task_desc

        while True:
            comm.Bcast([task_desc, MPI.INT], root=0)
            if task_desc[1] == MPI_MSG_TERMINATE:
                LOG.debug("Node %d received terminate message", comm_rank)
                return
            if task_desc[1] == MPI_MSG_CV:
                self._run_grid_search()
            elif task_desc[1] == MPI_MSG_TEST:
                self._run_train_test()
                break
            else:
                raise ValueError('unknown task with id %d' % task_desc[1])

        LOG.debug("Node %d is terminating", comm_rank)


class MPIGridSearchCVMaster(MPIBatchWorker):
    """Running on the root node and distributes work across slaves"""

    def __init__(self, param_grid, cv_iter, estimator, scorer, fit_params):
        super(MPIGridSearchCVMaster, self).__init__(estimator,
                                                    scorer, fit_params)
        self.param_grid = param_grid
        self.cv_iter = cv_iter

    def _create_batches(self):
        param_iter = ParameterGrid(self.param_grid)

        # divide work into batches equal to the communicator's size
        work_batches = [[] for _ in range(comm_size)]
        i = 0
        for fold_id, (train_index, test_index) in enumerate(self.cv_iter):
            for parameters in param_iter:
                work_batches[i % comm_size].append((fold_id + 1, train_index, test_index, parameters))
                i += 1

        return work_batches

    def _scatter_work(self):
        work_batches = self._create_batches()

        LOG.debug("Distributed items into %d batches of size %d", comm_size, len(work_batches[0]))

        # Distribute batches across all nodes
        root_work_batch = comm.scatter(work_batches, root=0)
        # The root node also does receive one batch it has to process
        root_result_batch = self.process_batch(root_work_batch)
        return root_result_batch

    def _gather_work(self, root_result_batch):
        # collect results: list of list of dict of parameters and performance measures
        result_batches = comm.gather(root_result_batch, root=0)

        out = []
        for result_batch in result_batches:
            if result_batch is None:
                continue
            for result_item in result_batch:
                out.append(result_item)
        LOG.debug("Received %d valid results", len(out))

        return pandas.DataFrame(out)

    def run(self, train_X, train_y):
        # tell slave that it should do hyper-parameter search
        self._task_desc[0] = 0
        self._task_desc[1] = MPI_MSG_CV

        comm.Bcast([self._task_desc, MPI.INT], root=0)
        comm.bcast((train_X, train_y), root=0)

        self._data_X = train_X
        self._data_y = train_y

        root_result_batch = self._scatter_work()
        return self._gather_work(root_result_batch)


def _fit_and_score_with_parameters(X, y, cv, best_parameters):
    """Distributes work of non-nested cross-validation across slave nodes"""

    # tell slaves testing phase is next
    _task_desc = numpy.empty(2, dtype=int)
    _task_desc[1] = MPI_MSG_TEST

    comm.Bcast([_task_desc, MPI.INT], root=0)
    comm.bcast((X, y), root=0)

    assert comm_size >= len(cv)

    for i, (train_index, test_index) in enumerate(cv):
        fold_id = i + 1
        LOG.info("Testing fold %d", fold_id)

        parameters = best_parameters.loc[fold_id, :].to_dict()
        work_item = (fold_id, train_index, test_index, parameters)

        comm.send(work_item, dest=fold_id, tag=MPI_TAG_TRAIN_TEST_DATA)

    scores = {}
    for i in range(len(cv)):
        fold_id, test_result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI_TAG_RESULT)
        scores[fold_id] = test_result

    # Tell all nodes to terminate
    for i in range(len(cv), comm_size):
        comm.send((0, None), dest=i, tag=MPI_TAG_TRAIN_TEST_DATA)

    return pandas.Series(scores)


class NestedGridSearchCV(BaseEstimator):
    """Cross-validation with nested hyper-parameter search for each training fold.

    The data is first split into ``cv`` train and test sets. For each training set.
    a grid search over the specified set of parameters is performed (inner cross-validation).
    The set of parameters that achieved the highest average score across all inner folds
    is used to re-fit a model on the entire training set of the outer cross-validation loop.
    Finally, results on the test set of the outer loop are reported.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        See sklearn.metrics.get_scorer for details.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    cv : integer or cross-validation generator, default=3
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    inner_cv : integer or callable, default=3
        If an integer is passed, it is the number of folds.
        If callable, the function must have the signature ``inner_cv_func(X, y)``
        and return a cross-validation object, see sklearn.cross_validation
        module for the list of possible objects.

    multi_output : boolean, default=False
        Allow multi-output y, as for multivariate regression.

    Attributes
    ----------
    best_params_ : pandas.DataFrame
        Contains selected parameter settings for each fold.
        The validation score refers to average score across all folds of the
        inner cross-validation, the test score to the score on the test set
        of the outer cross-validation loop.

    grid_scores_ : list of pandas.DataFrame
        Contains full results of grid search for each training set of the
        outer cross-validation loop.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, cv=None,
                 inner_cv=None, multi_output=False):
        self.scoring = scoring
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.fit_params = fit_params
        self.cv = cv
        self.inner_cv = inner_cv
        self.multi_output = multi_output

    def _grid_search(self, train_X, train_y):
        if callable(self.inner_cv):
            inner_cv = self.inner_cv(train_X, train_y)
        else:
            inner_cv = _check_cv(self.inner_cv, train_X, train_y, classifier=is_classifier(self.estimator))

        master = MPIGridSearchCVMaster(self.param_grid, inner_cv, self.estimator, self.scorer_, self.fit_params)
        return master.run(train_X, train_y)

    def _fit_master(self, X, y, cv):
        param_names = list(self.param_grid.keys())

        best_parameters = []
        grid_search_results = []
        for i, (train_index, test_index) in enumerate(cv):
            LOG.info("Training fold %d", i + 1)

            train_X = X[train_index, :]
            train_y = y[train_index]

            grid_results = self._grid_search(train_X, train_y)
            grid_search_results.append(grid_results)

            max_performance = _get_best_parameters(grid_results, param_names)
            LOG.info("Best performance for fold %d:\n%s", i + 1, max_performance)
            max_performance['fold'] = i + 1
            best_parameters.append(max_performance)

        best_parameters = pandas.DataFrame(best_parameters)
        best_parameters.set_index('fold', inplace=True)
        best_parameters['score (Test)'] = 0.0
        best_parameters.rename(columns={'score': 'score (Validation)'}, inplace=True)

        scores = _fit_and_score_with_parameters(X, y, cv, best_parameters.loc[:, param_names])
        best_parameters['score (Test)'] = scores

        self.best_params_ = best_parameters
        self.grid_scores_ = grid_search_results

    def _fit_slave(self):
        slave = MPISlave(self.estimator, self.scorer_, self.fit_params)
        slave.run()

    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=False, multi_output=self.multi_output)
        _check_param_grid(self.param_grid)

        cv = _check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if comm_rank == 0:
            self._fit_master(X, y, cv)
        else:
            self._fit_slave()

        return self

