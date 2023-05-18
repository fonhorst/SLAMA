from typing import Tuple, Union, Callable

import optuna
from lightautoml.ml_algo.tuning.optuna import OptunaTuner, TunableAlgo
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.validation.base import TrainValidIterator
from pyspark.sql import functions as sf

from examples.spark.examples_utils import get_spark_session
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.validation.iterators import SparkHoldoutIterator


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num)
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num)

    train_dataset, test_dataset = dataset.empty(), dataset.empty()
    train_dataset.set_data(train, dataset.features)
    test_dataset.set_data(test, dataset.features)

    return train_dataset, test_dataset


if __name__ == "__main__":
    spark = get_spark_session()

    # load data
    ds = SparkDataset()

    train_ds, test_ds = train_test_split(ds)

    # create main entities
    iterator = SparkHoldoutIterator(train_ds)
    tuner = OptunaTuner(n_trials=10, timeout=300)
    ml_algo = SparkBoostLGBM()
    score = ds.task.get_dataset_metric()

    # fit and predict
    model, oof_preds = tune_and_fit_predict(ml_algo, tuner, iterator)
    test_preds = ml_algo.predict(test_ds)

    #TODO: report trials
    # tuner.study.trials

    # estimate oof and test metrics
    oof_metric_value = score(oof_preds.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    test_metric_value = score(test_preds.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    print(f"OOF metric: {oof_metric_value}")
    print(f"Test metric: {oof_metric_value}")
