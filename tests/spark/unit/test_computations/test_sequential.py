import collections
import threading
from copy import deepcopy

import pytest
from pyspark.sql import SparkSession

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset

K = 20


class TestWorkerException(Exception):
    def __init__(self, id: int):
        super(TestWorkerException, self).__init__(f"Intentional exception in task {id}")


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


def test_allocate(spark: SparkSession, dataset: SparkDataset):
    manager = SequentialComputationsManager()

    # TODO: add checking for access from different threads
    with manager.session(dataset) as session:
        for i in range(10):
            with session.allocate() as slot:
                assert slot.dataset is not None
                assert slot.dataset.uid == dataset.uid

        acc = collections.deque()
        results = session.compute([build_func(acc, j) for j in range(K)])
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == 1
        assert next(iter(unique_thread_ids)) == threading.get_ident()

        acc = collections.deque()
        results = session.map_and_compute(build_idx_func(acc), list(range(K)))
        unique_thread_ids = set(acc)
        assert results == list(range(K))
        assert len(unique_thread_ids) == 1
        assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_compute():
    acc = collections.deque()
    tasks = [build_func(acc, i) for i in range(K)]

    manager = SequentialComputationsManager()
    results = manager.compute(tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def build_func_with_exception(acc: collections.deque, seq_id: int):
    def _func():
        acc.append(threading.get_ident())
        raise TestWorkerException(seq_id)
    return _func


def test_compute_with_exceptions(spark: SparkSession):
    acc = collections.deque()
    tasks = [*(build_func_with_exception(acc, i) for i in range(K, K + 3)), *(build_func(acc, i) for i in range(K))]

    manager = SequentialComputationsManager()
    with pytest.raises(TestWorkerException):
        manager.compute(tasks)


def test_compute_on_dataset(spark: SparkSession, dataset: SparkDataset):
    acc = collections.deque()
    tasks = [build_func_on_dataset(acc, i) for i in range(K)]

    manager = SequentialComputationsManager()
    results = manager.compute_on_dataset(dataset, tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_compute_on_train_val_iter(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = SequentialComputationsManager()
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_deepcopy(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = SequentialComputationsManager()

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()
