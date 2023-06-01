import collections
import itertools
import threading
from copy import deepcopy

import pytest
from pyspark.sql import SparkSession

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset

K = 20

# manager_configs = itertools.product([1, 2, 5], [False, True])
manager_configs = [(1, False)]


def build_func(acc: collections.deque, seq_id: int):
    def _func() -> int:
        acc.append(threading.get_ident())
        return seq_id
    return _func


def build_func_on_dataset(acc: collections.deque, seq_id: int):
    def _func(slot: ComputationSlot) -> int:
        assert slot.dataset is not None
        acc.append(threading.get_ident())
        return seq_id
    return _func


def build_fold_func(acc: collections.deque):
    def _func(fold_id: int, slot: ComputationSlot) -> int:
        assert slot.dataset is not None
        acc.append(threading.get_ident())
        return fold_id
    return _func


def build_idx_func(acc: collections.deque):
    def _func(idx: int) -> int:
        acc.append(threading.get_ident())
        return idx
    return _func


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_allocate(spark: SparkSession, dataset: SparkDataset, parallelism: int, use_location_prefs_mode: bool):
    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)

    # TODO: add checking for access from different threads
    with manager.session(dataset) as session:
        for i in range(10):
            with session.allocate() as slot:
                assert slot.dataset is not None
                if use_location_prefs_mode:
                    assert slot.dataset.uid != dataset.uid
                else:
                    assert slot.dataset.uid == dataset.uid

        acc = collections.deque()
        results = session.compute([build_func(acc, j) for j in range(K)])
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == min(K, parallelism)

        acc = collections.deque()
        results = session.map_and_compute(build_idx_func(acc), list(range(K)))
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute(parallelism: int, use_location_prefs_mode: bool):
    acc = collections.deque()
    tasks = [build_func(acc, i) for i in range(K)]

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute(tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute_on_dataset(spark: SparkSession, dataset: SparkDataset,
                            parallelism: int, use_location_prefs_mode: bool):
    acc = collections.deque()
    tasks = [build_func_on_dataset(acc, i) for i in range(K)]

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute_on_dataset(dataset, tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute_on_train_val_iter(spark: SparkSession, dataset: SparkDataset,
                                   parallelism: int, use_location_prefs_mode: bool):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)


def test_deepcopy(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    parallelism = 5
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = ParallelComputationsManager(parallelism=parallelism, use_location_prefs_mode=True)

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)
