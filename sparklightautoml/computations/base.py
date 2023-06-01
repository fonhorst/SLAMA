import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Any, Union, Dict
from typing import TypeVar, List

from sparklightautoml.computations.utils import deecopy_tviter_without_dataset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ComputationSlot:
    id: str
    dataset: Optional[SparkDataset] = None
    num_tasks: Optional[int] = None
    num_threads_per_executor: Optional[int] = None


class ComputationsSession(ABC):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def allocate(self) -> ComputationSlot:
        """
        Thread safe method
        Returns:

        """
        ...

    def map_and_compute(self, func: Callable[[R], T], tasks: List[R]) -> List[T]:
        ...

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        ...


class ComputationsManager(ABC):
    @property
    @abstractmethod
    def parallelism(self) -> int:
        ...

    @contextmanager
    @abstractmethod
    def session(self, dataset: Optional[SparkDataset] = None) -> ComputationsSession:
        ...

    def compute_on_dataset(self, dataset: SparkDataset, tasks: List[Callable[[ComputationSlot], T]]) -> List[T]:
        with self.session(dataset) as session:
            def _task_wrap(task):
                with session.allocate() as slot:
                    return task(slot)
            return session.map_and_compute(_task_wrap, tasks)

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        # use session here for synchronization purpose
        with self.session() as session:
            return session.compute(tasks)

    def compute_folds(self, train_val_iter: SparkBaseTrainValidIterator, task: Callable[[int, ComputationSlot], T])\
            -> List[T]:
        tv_iter = deecopy_tviter_without_dataset(train_val_iter)

        with self.session(train_val_iter.train) as session:
            def _task_wrap(fold_id: int):
                with session.allocate() as slot:
                    local_tv_iter = deepcopy(tv_iter)
                    local_tv_iter.train = slot.dataset
                    slot = deepcopy(slot)
                    slot.dataset = local_tv_iter[fold_id]
                    return task(fold_id, slot)

            fold_ids = list(range(len(train_val_iter)))
            return session.map_and_compute(_task_wrap, fold_ids)


# either parallelism settings or manager
ComputationsSettings = Union[Dict[str, Any], ComputationsManager]
