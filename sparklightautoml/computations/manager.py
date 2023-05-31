import logging
import math
import multiprocessing
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Callable, Optional, Iterable, Any, Dict, Tuple, Union
from typing import TypeVar, List

from pyspark import inheritable_thread_target, SparkContext, keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

T = TypeVar("T")
S = TypeVar("S", bound='Slot')

# either named profile and parallelism or parallelism settings or factory
AutoMLComputationsSettings = Union[Tuple[str, int], Dict[str, Any], 'ComputationManagerFactory']
# either parallelism settings or manager
ComputationsSettings = Union[Dict[str, Any], 'ComputationalJobManager']


class WorkloadType(Enum):
    ml_pipelines = "ml_pipelines"
    ml_algos = "ml_algos"
    job = "job"


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


def build_named_parallelism_settings(config_name: str, parallelism: int):
    parallelism_config = {
        "no_parallelism": {},
        "intra_mlpipe_parallelism": {
            "ml_pipelines": {"parallelism": 1},
            "ml_algos": {"parallelism": 1},
            "selector": {"parallelism": parallelism},
            "tuner": {"parallelism": parallelism},
            "linear_l2": {"parallelism": parallelism},
            "lgb": {
                "parallelism": parallelism,
                "use_location_prefs_mode": False
            }
        },
        "intra_mlpipe_parallelism_with_location_pres_mode": {
            "ml_pipelines": {"parallelism": 1},
            "ml_algos": {"parallelism": 1},
            "selector": {"parallelism": parallelism},
            "tuner": {"parallelism": parallelism},
            "linear_l2": {"parallelism": parallelism},
            "lgb": {
                "parallelism": parallelism,
                "use_location_prefs_mode": True
            }
        },
        "mlpipe_level_parallelism": {
            "ml_pipelines": {"parallelism": parallelism},
            "ml_algos": {"parallelism": 1},
            "selector": {"parallelism": 1},
            "tuner": {"parallelism": 1},
            "linear_l2": {"parallelism": 1},
            "lgb": {"parallelism": 1}
        }
    }

    assert config_name in parallelism_config, \
        f"Not supported parallelism mode: {config_name}. " \
        f"Only the following ones are supoorted at the moment: {list(parallelism_config.keys())}"

    return parallelism_config[config_name]


def build_computations_manager(computations_settings: Optional[ComputationsSettings] = None) \
        -> 'ComputationsManager':
    if computations_settings is not None and isinstance(computations_settings, ComputationsManager):
        computations_manager = computations_settings
    elif computations_settings is not None:
        assert isinstance(computations_settings, dict)
        parallelism = int(computations_settings.get('parallelism', '1'))
        use_location_prefs_mode = computations_settings.get('use_location_prefs_mode', False)
        computations_manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    else:
        computations_manager = SequentialComputationsManager()

    return computations_manager


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


class ComputationsManagerFactory:
    def __init__(self, computations_settings: Optional[Tuple[str, int], Dict[str, Any]] = None):
        super(ComputationsManagerFactory, self).__init__()
        computations_settings = computations_settings or ("no_parallelism", -1)

        if isinstance(computations_settings, Tuple):
            mode, parallelism = computations_settings
            self._computations_settings = build_named_parallelism_settings(mode, parallelism)
        else:
            self._computations_settings = computations_settings

        self._ml_pipelines_params = self._computations_settings.get('ml_pipelines', None)
        self._ml_algos_params = self._computations_settings.get('ml_algos', None)
        self._selector_params = self._computations_settings.get('selector', None)
        self._tuner_params = self._computations_settings.get('tuner', None)
        self._linear_l2_params = self._computations_settings.get('linear_l2', None)
        self._lgb_params = self._computations_settings.get('lgb', None)

    def get_ml_pipelines_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._ml_pipelines_params)

    def get_ml_algo_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._ml_algos_params)

    def get_selector_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._selector_params)

    def get_tuning_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._tuner_params)

    def get_lgb_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._lgb_params)

    def get_linear_manager(self) -> 'ComputationsManager':
        return build_computations_manager(self._linear_l2_params)


@dataclass
class ComputationSlot:
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
        self._session_lock = threading.Lock()

    def parallelism(self) -> int:
        return 1

    @contextmanager
    def session(self, dataset: Optional[SparkDataset] = None):
        with self._session_lock:
            self._dataset = dataset
            yield
            self._dataset = None

    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputationSlot], T]) \
            -> List[T]:
        return [task(i, ComputationSlot(train)) for i, train in enumerate(train_val_iter)]

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return [task() for task in tasks]

    def compute_on_dataset(self, dataset: SparkDataset, tasks: List[Callable[[ComputationSlot], T]]) -> List[T]:
        return [task(ComputationSlot(dataset)) for task in tasks]

    @contextmanager
    def allocate(self) -> ComputationSlot:
        assert self._dataset is not None, "Cannot allocate slots without session"
        yield ComputationSlot(self._dataset)


class ParallelComputationsManager(ComputationsManager):
    def __init__(self, parallelism: int = 1, use_location_prefs_mode: bool = False):
        # TODO: PARALLEL - accept a thread pool coming from above
        # doing it, because ParallelComputations Manager should be deepcopy-able
        # create_pools(1, 1, parallelism)
        assert parallelism >= 1
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._available_computing_slots_queue: Optional[Queue] = None
        self._pool = ThreadPool(processes=parallelism)
        self._session_lock = threading.Lock()

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @contextmanager
    def session(self, dataset: Optional[SparkDataset] = None):
        # TODO: PARALLEL - make this method thread safe by thread locking
        # TODO: PARALLEL - add id to slots
        with self._session_lock:
            with self._make_computing_slots(dataset) as computing_slots:
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
            yield [ComputationSlot(dataset) for _ in range(self._parallelism)]

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
                dataset=coalesced_dataset,
                num_tasks=num_tasks,
                num_threads_per_executor=num_threads_per_executor
            ))

            logger.debug(f"Preffered locations for slot #{i}: {pref_locs}")

        return dataset_slots

    def _map_and_compute(self, func: Callable[[], T], tasks: List[Any]) -> List[T]:
        return self._pool.map(inheritable_thread_target_with_exceptions_catcher(func), tasks)

    def _compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return self._pool.map(inheritable_thread_target_with_exceptions_catcher(lambda f: f()), tasks)


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
        # TODO: PARALLEL - incorrect implementation
        return self._tviter[fold_id]

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
