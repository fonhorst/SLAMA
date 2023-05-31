import logging
import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Callable, Optional, Any, Union, Dict
from typing import TypeVar, List

from sparklightautoml.computations.utils import get_executors, get_executors_cores, \
    inheritable_thread_target_with_exceptions_catcher, deecopy_tviter_without_dataset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.transformers.scala_wrappers.preffered_locs_partition_coalescer import \
    PrefferedLocsPartitionCoalescerTransformer
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S", bound='Slot')

# either parallelism settings or manager
ComputationsSettings = Union[Dict[str, Any], 'ComputationalJobManager']


@dataclass
class ComputationSlot:
    id: str
    dataset: Optional[SparkDataset] = None
    num_tasks: Optional[int] = None
    num_threads_per_executor: Optional[int] = None


class ComputationsManager(ABC):
    @property
    @abstractmethod
    def parallelism(self) -> int:
        ...

    @contextmanager
    @abstractmethod
    def session(self, dataset: Optional[SparkDataset] = None):
        ...

    @abstractmethod
    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputationSlot], T])\
            -> List[T]:
        ...

    @abstractmethod
    def compute_on_dataset(self, dataset: SparkDataset, tasks: List[Callable[[ComputationSlot], T]]) \
            -> List[T]:
        ...

    @abstractmethod
    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        ...

    @abstractmethod
    def allocate(self) -> ComputationSlot:
        """
        Thread safe method
        Returns:

        """
        ...


class SequentialComputationsManager(ComputationsManager):
    def __init__(self):
        super(SequentialComputationsManager, self).__init__()
        self._dataset: Optional[SparkDataset] = None

    def parallelism(self) -> int:
        return 1

    @contextmanager
    def session(self, dataset: Optional[SparkDataset] = None):
        self._dataset = dataset
        yield
        self._dataset = None

    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputationSlot], T]) \
            -> List[T]:
        return [task(i, ComputationSlot(f"{i}", train)) for i, train in enumerate(train_val_iter)]

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return [task() for task in tasks]

    def compute_on_dataset(self, dataset: SparkDataset, tasks: List[Callable[[ComputationSlot], T]]) -> List[T]:
        return [task(ComputationSlot("0", dataset)) for task in tasks]

    @contextmanager
    def allocate(self) -> ComputationSlot:
        assert self._dataset is not None, "Cannot allocate slots without session"
        yield ComputationSlot("0", self._dataset)

    def __deepcopy__(self, memodict={}):
        assert self._dataset is None, "Cannot deepcopying during session"
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result


class ParallelComputationsManager(ComputationsManager):
    def __init__(self, parallelism: int = 1, use_location_prefs_mode: bool = False):
        assert parallelism >= 1
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._available_computing_slots_queue: Optional[Queue] = None
        self._pool: Optional[ThreadPool] = None

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @contextmanager
    def session(self, dataset: Optional[SparkDataset] = None):
        with self._obtain_thread_pool(), self._make_computing_slots(dataset) as computing_slots:
            self._available_computing_slots_queue = Queue(maxsize=len(computing_slots))
            for cslot in computing_slots:
                self._available_computing_slots_queue.put(cslot)
            yield
            self._available_computing_slots_queue = None

    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputationSlot], T])\
            -> List[T]:
        tv_iter = deecopy_tviter_without_dataset(train_val_iter)

        with self.session(train_val_iter.train):
            def _task_wrap(fold_id: int):
                with self.allocate() as slot:
                    local_tv_iter = deepcopy(tv_iter)
                    local_tv_iter.train = slot.dataset
                    slot = deepcopy(slot)
                    slot.dataset = local_tv_iter[fold_id]
                    return task(fold_id, slot)

            fold_ids = list(range(len(train_val_iter)))
            return self._map_and_compute(_task_wrap, fold_ids)

    def compute_on_dataset(self, dataset: SparkDataset, tasks: List[Callable[[ComputationSlot], T]]) -> List[T]:
        with self.session(dataset):
            def _task_wrap(task):
                with self.allocate() as slot:
                    return task(slot)
            return self._map_and_compute(_task_wrap, tasks)

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        # use session here for synchronization purpose
        with self.session():
            return self._compute(tasks)

    @contextmanager
    def allocate(self) -> ComputationSlot:
        assert self._available_computing_slots_queue is not None, "Cannot allocate slots without session"
        slot = self._available_computing_slots_queue.get()
        yield slot
        self._available_computing_slots_queue.put(slot)

    @contextmanager
    def _make_computing_slots(self, dataset: Optional[SparkDataset]) -> List[ComputationSlot]:
        if dataset is not None and self._use_location_prefs_mode:
            computing_slots = None
            try:
                computing_slots = self._coalesced_dataset_copies_into_preffered_locations(dataset)
                yield computing_slots
            finally:
                if computing_slots is not None:
                    for cslot in computing_slots:
                        cslot.dataset.unpersist()
        else:
            yield [ComputationSlot(f"i", dataset) for i in range(self._parallelism)]

    @contextmanager
    def _obtain_thread_pool(self):
        try:
            self._pool = ThreadPool(processes=self._parallelism)
            yield
        finally:
            self._pool.terminate()
            self._pool = None

    def _coalesced_dataset_copies_into_preffered_locations(self, dataset: SparkDataset) \
            -> List[ComputationSlot]:
        logger.warning("Be aware for correct functioning slot-based computations "
                       "there should noy be any parallel computations from "
                       "different entities (other MLPipes, MLAlgo, etc).")

        # TODO: PARALLEL - improve function to work with uneven number of executors
        execs = get_executors()
        exec_cores = get_executors_cores()
        execs_per_slot = max(1, math.floor(len(execs) / self._parallelism))
        slots_num = int(len(execs) / execs_per_slot)
        num_tasks = execs_per_slot * exec_cores
        num_threads_per_executor = max(exec_cores - 1, 1)

        if len(execs) % self._parallelism != 0:
            warnings.warn(f"Uneven number of executors per job. "
                          f"Setting execs per slot: {execs_per_slot}, slots num: {slots_num}.")

        logger.info(f"Coalescing dataset into multiple copies (num copies: {slots_num}) "
                    f"with specified preffered locations")

        dataset_slots = []

        # TODO: PARALLEL - may be executed in parallel
        # TODO: PARALLEL - it might be optimized on Scala level and squashed into a single operation
        for i in range(slots_num):
            pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]

            coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs)\
                .transform(dataset.data).cache()
            coalesced_data.write.mode('overwrite').format('noop').save()

            coalesced_dataset = dataset.empty()
            coalesced_dataset.set_data(coalesced_data, coalesced_dataset.features, coalesced_dataset.roles,
                                       name=f"CoalescedForPrefLocs_{dataset.name}")

            dataset_slots.append(ComputationSlot(
                id=f"{i}",
                dataset=coalesced_dataset,
                num_tasks=num_tasks,
                num_threads_per_executor=num_threads_per_executor
            ))

            logger.debug(f"Preffered locations for slot #{i}: {pref_locs}")

        return dataset_slots

    def _map_and_compute(self, func: Callable[[], T], tasks: List[Any]) -> List[T]:
        assert self._pool is not None
        return self._pool.map(inheritable_thread_target_with_exceptions_catcher(func), tasks)

    def _compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        assert self._pool is not None
        return self._pool.map(inheritable_thread_target_with_exceptions_catcher(lambda f: f()), tasks)

    def __deepcopy__(self, memodict={}):
        assert self._pool is None, "Cannot deepcopying during session"
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result
