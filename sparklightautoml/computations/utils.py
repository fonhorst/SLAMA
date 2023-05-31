import multiprocessing
from copy import deepcopy
from typing import List, Iterable, Optional

from pyspark import SparkContext, inheritable_thread_target
from pyspark.sql import SparkSession

from sparklightautoml.computations.managers import logger, ComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator


def get_executors() -> List[str]:
    # noinspection PyUnresolvedReferences
    sc = SparkContext._active_spark_context
    return sc._jvm.org.apache.spark.lightautoml.utils.SomeFunctions.executors()


def get_executors_cores() -> int:
    master_addr = SparkSession.getActiveSession().conf.get("spark.master")
    if master_addr.startswith("local-cluster"):
        _, cores_str, _ = master_addr[len("local-cluster["): -1].split(",")
        cores = int(cores_str)
    elif master_addr.startswith("local"):
        cores_str = master_addr[len("local["): -1]
        cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
    else:
        cores = int(SparkSession.getActiveSession().conf.get("spark.executor.cores", "1"))

    return cores


def inheritable_thread_target_with_exceptions_catcher(f):
    def _func():
        try:
            return f()
        except:
            logger.error("Error in a compute thread", exc_info=True)
            raise

    return inheritable_thread_target(_func)


def deecopy_tviter_without_dataset(tv_iter: SparkBaseTrainValidIterator) -> SparkBaseTrainValidIterator:
    train = tv_iter.train
    tv_iter.train = None
    tv_iter_copy = deepcopy(tv_iter)
    tv_iter.train = train
    return tv_iter_copy


class _SlotInitiatedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, computations_manager: ComputationsManager, tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._computations_manager = computations_manager
        self._tviter = deecopy_tviter_without_dataset(tviter)

    def __iter__(self) -> Iterable:
        def _iter():
            with self._computations_manager.allocate() as slot:
                self._tviter.train = slot.dataset
                for elt in self._tviter:
                    yield elt
                self._tviter = None

        return _iter()

    def __len__(self) -> Optional[int]:
        return len(self._tviter)

    def __getitem__(self, fold_id: int) -> SparkDataset:
        with self._computations_manager.allocate() as slot:
            self._tviter.train = slot.dataset
            dataset = self._tviter[fold_id]
            self._tviter = None
        return dataset

    def __next__(self):
        raise NotImplementedError("NotSupportedMethod")

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError("NotSupportedMethod")

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError("NotSupportedMethod")

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()

    def convert_to_holdout_iterator(self):
        return _SlotInitiatedTVIter(self._computations_manager, self._tviter.convert_to_holdout_iterator())