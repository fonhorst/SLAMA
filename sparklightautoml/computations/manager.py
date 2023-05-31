import logging
import math
import multiprocessing
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
# either parallelism degree or manager
ComputationsStagesSettings = Union[int, 'AutoMLStageManager']
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
        "no_parallelism": None,
        "intra_mlpipe_parallelism": {
            "ml_pipelines": 1,
            "ml_algos": 1,
            "selector": parallelism,
            "tuner": parallelism,
            "linear_l2": None,
            "lgb": {
                "parallelism": parallelism,
                "use_location_prefs_mode": False
            }
        },
        "intra_mlpipe_parallelism_with_experimental_features": {
            "ml_pipelines": 1,
            "ml_algos": 1,
            "selector": parallelism,
            "tuner": parallelism,
            "linear_l2": None,
            "lgb": {
                "parallelism": parallelism,
                "use_location_prefs_mode": True
            }
        },
        "mlpipe_level_parallelism": {
            "ml_pipelines": parallelism,
            "ml_algos": 1,
            "selector": 1,
            "tuner": 1,
            "linear_l2": None,
            "lgb": None
        }
    }

    assert config_name in parallelism_config, \
        f"Not supported parallelism mode: {config_name}. " \
        f"Only the following ones are supoorted at the moment: {list(parallelism_config.keys())}"

    return parallelism_config[config_name]


class ComputationManagerFactory:
    def __init__(self, computations_settings: Optional[Tuple[str, int], Dict[str, Any]] = None):
        super(ComputationManagerFactory, self).__init__()
        computations_settings = computations_settings or ("no_parallelism", -1)

        if isinstance(computations_settings, Tuple):
            mode, parallelism = computations_settings
            self._computations_settings = build_named_parallelism_settings(mode, parallelism)
        else:
            self._computations_settings = computations_settings

        self._ml_pipelines_parallelism = int(self._computations_settings.get('ml_pipelines', '1'))
        self._ml_algos_parallelism = int(self._computations_settings.get('ml_algos', '1'))
        self._selector_parallelism = int(self._computations_settings.get('selector', '1'))
        self._tuner_parallelism = int(self._computations_settings.get('tuner', '1'))
        self._linear_l2_params = self._computations_settings.get('linear_l2', None)
        self._lgb_params = self._computations_settings.get('lgb', None)

    def get_ml_pipelines_manager(self) -> 'AutoMLStageManager':
        return build_computations_stage_manager(self._ml_pipelines_parallelism)

    def get_ml_algo_manager(self) -> 'AutoMLStageManager':
        return build_computations_stage_manager(self._ml_algos_parallelism)

    def get_selector_manager(self) -> 'AutoMLStageManager':
        return build_computations_stage_manager(self._selector_parallelism)

    def get_tuning_manager(self) -> 'ComputationalJobManager':
        return build_computations_manager({"parallelism": self._tuner_parallelism})

    def get_lgb_manager(self) -> 'ComputationalJobManager':
        return build_computations_manager(self._lgb_params)

    def get_linear_manager(self) -> 'ComputationalJobManager':
        return build_computations_manager(self._linear_l2_params)


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
    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputingSlot], T])\
            -> List[T]:
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


class SequentialComputationalJobManager(ComputationalJobManager):
    def __init__(self):
        super(SequentialComputationalJobManager, self).__init__()
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
        # create_pools(1, 1, parallelism)
        assert parallelism >= 1
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._available_computing_slots_queue: Optional[Queue] = None
        self._pool = ThreadPool(processes=parallelism)

    def session(self, dataset: SparkDataset):
        # TODO: PARALLEL - add id to slots
        with self._make_computing_slots(dataset) as computing_slots:
            self._available_computing_slots_queue = Queue(maxsize=len(computing_slots))
            for cslot in computing_slots:
                self._available_computing_slots_queue.put(cslot)
            yield
            self._available_computing_slots_queue = None

    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputingSlot], T])\
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
            return self._map(_task_wrap, fold_ids)

    def compute(self, dataset: SparkDataset, tasks: List[Callable[[ComputingSlot], T]]) -> List[T]:
        with self.session(dataset):
            def _task_wrap(task):
                with self.allocate() as slot:
                    return task(slot)
            return self._map(_task_wrap, tasks)

    @contextmanager
    def allocate(self) -> ComputingSlot:
        assert self._available_computing_slots_queue is not None, "Cannot allocate slots without session"
        slot = self._available_computing_slots_queue.get()
        yield slot
        self._available_computing_slots_queue.put(slot)

    @contextmanager
    def _make_computing_slots(self, dataset) -> List[ComputingSlot]:
        if self._use_location_prefs_mode:
            computing_slots = None
            try:
                computing_slots = self._coalesced_dataset_copies_into_preffered_locations(dataset)
                yield computing_slots
            finally:
                if computing_slots is not None:
                    for cslot in computing_slots:
                        cslot.dataset.unpersist()
        else:
            yield [ComputingSlot(dataset) for _ in range(self._parallelism)]

    def _coalesced_dataset_copies_into_preffered_locations(self, dataset: SparkDataset) \
            -> List[ComputingSlot]:
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

            dataset_slots.append(ComputingSlot(
                dataset=coalesced_dataset,
                num_tasks=num_tasks,
                num_threads_per_executor=num_threads_per_executor
            ))

            logger.debug(f"Preffered locations for slot #{i}: {pref_locs}")

        return dataset_slots

    def _map(self, func: Callable[[], T], tasks: List[Any]) -> List[T]:
        return self._pool.map(inheritable_thread_target_with_exceptions_catcher(func), tasks)


class SequentialAutoMLStageManager(AutoMLStageManager):
    def __init__(self):
        super(SequentialAutoMLStageManager, self).__init__()

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return [task() for task in tasks]


class ParallelAutoMLStageManager(AutoMLStageManager):
    def __init__(self, parallelism: int = 1):
        super(ParallelAutoMLStageManager, self).__init__()
        self._pool = ThreadPool(processes=parallelism)

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return self._pool.map(lambda task: task(), tasks)


def build_computations_manager(computations_settings: Optional[ComputationsSettings] = None) -> ComputationalJobManager:
    if computations_settings is not None and isinstance(computations_settings, ComputationalJobManager):
        computations_manager = computations_settings
    elif computations_settings is not None:
        assert isinstance(computations_settings, dict)
        parallelism = int(computations_settings.get('parallelism', '1'))
        use_location_prefs_mode = computations_settings.get('use_location_prefs_mode', False)
        computations_manager = ParallelComputationalJobManager(parallelism, use_location_prefs_mode)
    else:
        computations_manager = SequentialComputationalJobManager()

    return computations_manager


def build_computations_stage_manager(computations_stage_settings: Optional[ComputationsStagesSettings] = None) -> AutoMLStageManager:
    if computations_stage_settings is not None and isinstance(computations_stage_settings, AutoMLStageManager):
        computations_manager = computations_stage_settings
    elif computations_stage_settings is not None:
        assert isinstance(computations_stage_settings, int) and computations_stage_settings >= 1
        computations_manager = ParallelAutoMLStageManager(parallelism=computations_stage_settings)
    else:
        computations_manager = SequentialAutoMLStageManager()

    return computations_manager


class _SlotInitiatedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, computations_manager: ComputationalJobManager, tviter: SparkBaseTrainValidIterator):
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
