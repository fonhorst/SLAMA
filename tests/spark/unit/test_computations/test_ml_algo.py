import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset


@pytest.mark.parametrize("manager,ml_algo", [
    (SequentialComputationsManager(), SparkBoostLGBM(use_single_dataset_mode=True, use_barrier_execution_mode=True))
])
def test_ml_algo(spark: SparkSession, dataset: SparkDataset, manager: ComputationsManager, ml_algo: SparkTabularMLAlgo):
    tv_iter = SparkFoldsIterator(dataset)

    ml_algo.computations_manager = manager

    oof_preds = ml_algo.fit_predict(tv_iter)
    preds = ml_algo.predict(dataset)

    assert ml_algo.prediction_feature in oof_preds.features
    assert ml_algo.prediction_feature in oof_preds.data.columns
    assert ml_algo.prediction_feature in preds.features
    assert ml_algo.prediction_feature in preds.data.columns

    assert oof_preds.data.count() == dataset.data.count()
    assert preds.data.count() == dataset.data.count()

    score = dataset.task.get_dataset_metric()
    oof_metric = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))
    test_metric = score(preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    assert oof_metric > 0
    assert test_metric > 0
