import logging
import math
import multiprocessing
import threading
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Callable, Optional, Iterable, Any, Dict, Tuple, cast, Union
from typing import TypeVar, List

from pyspark import inheritable_thread_target, SparkContext, keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, TrainVal

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

T = TypeVar("T")
S = TypeVar("S", bound='Slot')


ComputationsStagesSettings = Union[Tuple[str, int], Dict[str, Any], 'AutoMLStageManager']
AutoMLComputationsSettings = Union[Tuple[str, int], Dict[str, Any], 'ComputationManagerFactory']
ComputationsSettings = Union[Dict[str, Any], 'ComputationalJobManager']


class WorkloadType(Enum):
    ml_pipelines = "ml_pipelines"
    ml_algos = "ml_algos"
    job = "job"


__pools__: Optional[Dict[WorkloadType, ThreadPool]] = None


def create_pools(ml_pipes_pool_size: int = 1, ml_algos_pool_size: int = 1, job_pool_size: int = 1):
    global __pools__

    assert __pools__ is None, "Cannot recreate already existing thread pools"

    __pools__ = {
        WorkloadType.ml_pipelines: ThreadPool(processes=ml_pipes_pool_size) if ml_pipes_pool_size > 1 else None,
        WorkloadType.ml_algos: ThreadPool(processes=ml_algos_pool_size) if ml_algos_pool_size > 1 else None,
        WorkloadType.job: ThreadPool(processes=job_pool_size) if job_pool_size > 1 else None
    }


def get_pool(pool_type: WorkloadType) -> Optional[ThreadPool]:
    return __pools__.get(pool_type, None)


# noinspection PyUnresolvedReferences
def get_executors() -> List[str]:
    master_addr = SparkSession.getActiveSession().conf.get("spark.master")
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


def _compute_sequential(tasks: List[Callable[[], T]]) -> List[T]:
    return [task() for task in tasks]


def build_named_parallelism_settings(config_name: str, parallelism: int):
    parallelism_config = {
        "no_parallelism": None,
        "intra_mlpipe_parallelism": {
            WorkloadType.ml_pipelines.name: 1,
            WorkloadType.ml_algos.name: 1,
            WorkloadType.job.name: parallelism,
            "tuner": parallelism,
            "linear_l2": {},
            "lgb": {
                "parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "experimental_parallel_mode": False
            }
        },
        "intra_mlpipe_parallelism_with_experimental_features": {
            WorkloadType.ml_pipelines.name: 1,
            WorkloadType.ml_algos.name: 1,
            WorkloadType.job.name: parallelism,
            "tuner": parallelism,
            "linear_l2": {},
            "lgb": {
                "parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "experimental_parallel_mode": True
            }
        },
        "mlpipe_level_parallelism": {
            WorkloadType.ml_pipelines.name: parallelism,
            WorkloadType.ml_algos.name: 1,
            WorkloadType.job.name: 1,
            "tuner": 1,
            "linear_l2": {},
            "lgb": {}
        }
    }

    assert config_name in parallelism_config, \
        f"Not supported parallelism mode: {config_name}. " \
        f"Only the following ones are supoorted at the moment: {list(parallelism_config.keys())}"

    return parallelism_config[config_name]


@inherit_doc
class PrefferedLocsPartitionCoalescerTransformer(JavaTransformer):
    """
    Custom implementation of PySpark BalancedUnionPartitionsCoalescerTransformer wrapper
    """

    @keyword_only
    def __init__(self, pref_locs: List[str], do_shuffle: bool = True):
        super(PrefferedLocsPartitionCoalescerTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.lightautoml.utils.PrefferedLocsPartitionCoalescerTransformer",
            self.uid, pref_locs, do_shuffle
        )


@dataclass
class ComputingSlot:
    dataset: SparkDataset
    num_tasks: Optional[int] = None
    num_threads_per_executor: Optional[int] = None


@dataclass
class SlotSize:
    num_tasks: int
    num_threads_per_executor: int


class ComputingSession(ABC):
    @abstractmethod
    @contextmanager
    def allocate(self) -> ComputingSlot:
        ...


class ParallelSlotAllocator(ComputingSession):
    def __init__(self, slot_size: SlotSize, slots: List[ComputingSlot], pool: ThreadPool):
        assert len(slots) > 0
        self._slot_size = slot_size
        self._slots = slots
        self._pool = pool
        self._slots_lock = threading.Lock()

    @property
    def slot_size(self) -> SlotSize:
        return self._slot_size

    @contextmanager
    def allocate(self) -> ComputingSlot:
        try:
            while True:
                try:
                    with self._slots_lock:
                        free_slot = next((slot for slot in self._slots if slot.free))
                        free_slot.free = False
                    break
                except StopIteration:
                    logger.debug("No empty slot, sleeping and repeating again")
                time.sleep(5)

            yield free_slot

            with self._slots_lock:
                free_slot.free = True
        except Exception as ex:
            logger.error("Some bad exception happened", exc_info=1)
            raise ex


class AutoMLStageManager(ABC):
    @abstractmethod
    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        ...


class ComputationalJobManager(ABC):
    @contextmanager
    @abstractmethod
    def session(self, dataset: SparkDataset):
        ...

    @abstractmethod
    def compute(self, dataset: SparkDataset, tasks: List[Callable[[ComputingSlot], T]]) -> List[T]:
        ...

    @abstractmethod
    def allocate(self) -> ComputingSlot:
        """
        Thread safe method
        Returns:

        """
        ...


class ComputationManagerFactory:
    def __init__(self, computations_settings: Optional[Dict[str, Any]] = None):
        pass

    def get_ml_pipelines_manager(self) -> AutoMLStageManager:
        pass

    def get_ml_algo_manager(self) -> AutoMLStageManager:
        pass

    def get_gbm_manager(self) -> ComputationalJobManager:
        pass

    def get_linear_manager(self) -> ComputationalJobManager:
        pass

    def get_tuning_manager(self) -> ComputationalJobManager:
        pass

    def get_selector_manager(self) -> ComputationalJobManager:
        pass


class SequentialComputationalJobManager(ComputationalJobManager):
    def __init__(self, default_slot_size: Optional[SlotSize] = None):
        super(SequentialComputationalJobManager, self).__init__()
        self._default_slot_size = default_slot_size
        self._dataset: Optional[SparkDataset] = None

    @contextmanager
    def session(self, dataset: SparkDataset):
        self._dataset = dataset
        yield
        self._dataset = None

    def compute(self, dataset: SparkDataset, tasks: List[Callable[[ComputingSlot], T]]) -> List[T]:
        return [task(ComputingSlot(dataset)) for task in tasks]

    def allocate(self) -> ComputingSlot:
        assert self._dataset is not None, "Cannot allocate slots without session"
        return ComputingSlot(self._dataset)


class ParallelComputationalJobManager(ComputationalJobManager):
    def __init__(self, parallelism: int = 1, use_location_prefs_mode: bool = True):
        # doing it, because ParallelComputations Manager should be deepcopy-able
        create_pools(1, 1, parallelism)
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._available_computing_slots_queue: Optional[Queue] = None

    def session(self, dataset: SparkDataset):
        # TODO: PARALLEL - add id to slots
        with self._prepare_datasets(dataset) as computing_slots:
            self._available_computing_slots_queue = Queue(maxsize=len(computing_slots))
            for cslot in computing_slots:
                self._available_computing_slots_queue.put(cslot)
            yield
            self._available_computing_slots_queue = None

    def compute(self, dataset: SparkDataset, tasks: List[Callable[[ComputingSlot], T]]) -> List[T]:
        with self.session(dataset):
            pool = self._get_pool()

            # TODO: PARALLEL - Exception catcher should be added here
            # TODO: PARALLEL - Can be simplified
            ptasks = list(map(inheritable_thread_target, tasks))
            results = list(pool.imap_unordered(lambda f: f(self.allocate()), ptasks))
            results = sorted(
                (result for result in results if result),
                key=lambda x: x[0]
            )

        return results

    def allocate(self) -> ComputingSlot:
        return self._available_computing_slots_queue.get()

    def _prepare_dataset(self, dataset: SparkDataset) -> SparkDataset:
        raise NotImplementedError()


    def _get_pool(self, pool_type: WorkloadType) -> Optional[ThreadPool]:
        return get_pool(pool_type)

    @property
    def can_support_slots(self) -> bool:
        return True

    @property
    def default_slot_size(self) -> Optional[SlotSize]:
        return self._default_slot_size



        return results

    @contextmanager
    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: WorkloadType) -> ComputingSession:
        logger.warning("Beaware for correct functioning slot-based computations "
                       "there should noy be any parallel computations from "
                       "different entities (other MLPipes, MLAlgo, etc).")

        pool = self._get_pool(pool_type)
        slot_size, slots = self._prepare_trains(dataset=dataset, parallelism=parallelism)

        yield ParallelSlotAllocator(slot_size, slots, pool)

        logger.info("Clear cache of dataset copies (slots) for the coalesced dataset")
        for slot in slots:
            slot.dataset.data.unpersist()

    @staticmethod
    def _prepare_trains(dataset: SparkDataset, parallelism: int) -> Tuple[SlotSize, List[ComputingSlot]]:
        execs = get_executors()
        exec_cores = get_executors_cores()
        execs_per_slot = max(1, math.floor(len(execs) / parallelism))
        slots_num = int(len(execs) / execs_per_slot)
        slot_size = SlotSize(num_tasks=execs_per_slot * exec_cores, num_threads_per_executor=max(exec_cores - 1, 1))

        if len(execs) % parallelism != 0:
            warnings.warn(f"Uneven number of executors per job. "
                          f"Setting execs per slot: {execs_per_slot}, slots num: {slots_num}.")

        logger.info(f"Coalescing dataset into multiple copies (num copies: {slots_num}) "
                    f"with specified preffered locations")

        _train_slots = []

        for i in range(slots_num):
            pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]

            coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs)\
                .transform(dataset.data).cache()
            coalesced_data.write.mode('overwrite').format('noop').save()

            coalesced_dataset = dataset.empty()
            coalesced_dataset.set_data(coalesced_data, coalesced_dataset.features, coalesced_dataset.roles,
                                       name=f"CoalescedForPrefLocs_{dataset.name}")

            _train_slots.append(ComputingSlot(dataset=coalesced_dataset, free=True))

            logger.info(f"Preffered locations for slot #{i}: {pref_locs}")

        return slot_size, _train_slots

    @contextmanager
    def _prepare_datasets(self, dataset) -> List[ComputingSlot]:
        if self._use_location_prefs_mode:
            # TODO: PARALLEL - add try .. finally to clean computing slots datasets
            raise NotImplementedError()
            # TODO: PARALLEL - clean computing slots datasets
        else:
            yield [ComputingSlot(dataset) for _ in range(self._parallelism)]


def build_computations_manager(parallelism_settings: Optional[Dict[str, Any]] = None):
    if parallelism_settings is None:
        comp_manager = SequentialComputationsManagerComputational()
    else:
        comp_manager = ParallelComputationsManagerComputational(
            ml_pipes_pool_size=parallelism_settings[WorkloadType.ml_pipelines.name],
            ml_algos_pool_size=parallelism_settings[WorkloadType.ml_algos.name],
            job_pool_size=parallelism_settings[WorkloadType.job.name]
        )
    return comp_manager


def default_computations_manager() -> ComputationalJobManager:
    return SequentialComputationsManagerComputational()


class _SlotBasedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, slots: Callable[[], ComputingSlot], tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slots = slots
        self._tviter = tviter
        self._curr_pos = 0

    def __iter__(self) -> Iterable:
        self._curr_pos = 0
        return self

    def __len__(self) -> Optional[int]:
        return len(self._tviter)

    def __next__(self) -> TrainVal:
        with self._slots() as slot:
            tviter = deepcopy(self._tviter)
            tviter.train = slot.dataset

            self._curr_pos += 1

            try:
                curr_tv = None
                for i in range(self._curr_pos):
                    curr_tv = next(tviter)
            except StopIteration:
                self._curr_pos = 0
                raise StopIteration()

        return curr_tv

    def convert_to_holdout_iterator(self):
        return _SlotBasedTVIter(
            self._slots,
            cast(SparkBaseTrainValidIterator, self._tviter.convert_to_holdout_iterator())
        )

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError()

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError()

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()


class _SlotInitiatedTVIter(SparkBaseTrainValidIterator):
    def __len__(self) -> Optional[int]:
        return len(self._tviter)

    def convert_to_holdout_iterator(self):
        return _SlotInitiatedTVIter(self._slot_allocator, self._tviter.convert_to_holdout_iterator())

    def __init__(self, slot_allocator: ComputationalJobManager, tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slot_allocator = slot_allocator
        self._tviter = deepcopy(tviter)

    def __iter__(self) -> Iterable:
        def _iter():
            with self._slot_allocator.allocate() as slot:
                tviter = deepcopy(self._tviter)
                tviter.train = slot.dataset
                for elt in tviter:
                    yield elt

        return _iter()

    def __next__(self):
        raise NotImplementedError("NotSupportedMethod")

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError("NotSupportedMethod")

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError("NotSupportedMethod")

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()