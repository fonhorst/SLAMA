import logging
from copy import deepcopy
from logging import config
from typing import Tuple, Union, Callable

import optuna
from lightautoml.ml_algo.tuning.optuna import TunableAlgo
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import functions as sf

from examples.spark.examples_utils import get_spark_session
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.computations.utils import deecopy_tviter_without_dataset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.ml_algo.tuning.parallel_optuna import ParallelOptunaTuner
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class ProgressReportingOptunaTuner(ParallelOptunaTuner):
    def _get_objective(self,
                       ml_algo: TunableAlgo,
                       estimated_n_trials: int,
                       train_valid_iterator: SparkBaseTrainValidIterator) \
            -> Callable[[optuna.trial.Trial], Union[float, int]]:
        assert isinstance(ml_algo, SparkTabularMLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            with self._session.allocate() as slot:
                assert slot.dataset is not None
                _ml_algo = deepcopy(ml_algo)
                tv_iter = deecopy_tviter_without_dataset(train_valid_iterator)
                tv_iter.train = slot.dataset

                optimization_search_space = _ml_algo.optimization_search_space

                if not optimization_search_space:
                    optimization_search_space = _ml_algo._get_default_search_spaces(
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                        estimated_n_trials=estimated_n_trials,
                    )

                if callable(optimization_search_space):
                     params = optimization_search_space(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )
                else:
                    params = self._sample(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )

                _ml_algo.params = params

                logger.warning(f"Optuna Params: {params}")

                output_dataset = _ml_algo.fit_predict(train_valid_iterator=tv_iter)
                obj_score = _ml_algo.score(output_dataset)

                logger.info(f"Objective score: {obj_score}")
                return obj_score

        return objective


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num)
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num)

    train_dataset, test_dataset = dataset.empty(), dataset.empty()
    train_dataset.set_data(train, dataset.features, roles=dataset.roles)
    test_dataset.set_data(test, dataset.features, roles=dataset.roles)

    return train_dataset, test_dataset


if __name__ == "__main__":
    spark = get_spark_session()

    # feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
    feat_pipe = "linear"  # linear, lgb_simple or lgb_adv
    dataset_name = "lama_test_dataset"
    parallelism = 3

    # load and prepare data
    ds = SparkDataset.load(
        path=f"/tmp/{dataset_name}__{feat_pipe}__features.dataset",
        persistence_manager=PlainCachePersistenceManager()
    )
    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)

    # create main entities
    computations_manager = ParallelComputationsManager(parallelism=parallelism, use_location_prefs_mode=True)
    iterator = SparkFoldsIterator(train_ds).convert_to_holdout_iterator()
    tuner = ProgressReportingOptunaTuner(
        n_trials=10,
        timeout=300,
        parallelism=parallelism,
        computations_manager=computations_manager
    )
    # ml_algo = SparkBoostLGBM(default_params={"numIterations": 500}, computations_settings=computations_manager)
    ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]}, computations_settings=computations_manager)
    score = ds.task.get_dataset_metric()

    # fit and predict
    model, oof_preds = tune_and_fit_predict(ml_algo, tuner, iterator)
    test_preds = ml_algo.predict(test_ds)

    # estimate oof and test metrics
    oof_metric_value = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    test_metric_value = score(test_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    print(f"OOF metric: {oof_metric_value}")
    print(f"Test metric: {oof_metric_value}")

    spark.stop()
