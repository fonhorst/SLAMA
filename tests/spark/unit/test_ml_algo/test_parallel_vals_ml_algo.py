from typing import cast, Callable

import numpy as np
import pytest
from lightautoml.dataset.base import RolesDict
from pyspark.sql import SparkSession

from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess
from ..dataset_utils import get_test_datasets

spark = spark_sess

ml_alg_kwargs = {
    'auto_unique_co': 10,
    'max_intersection_depth': 3,
    'multiclass_te_co': 3,
    'output_categories': True,
    'top_intersections': 4
}


@pytest.mark.skip(reason="Not implemented yet")
def test_parallel_timer_exceeded(spark: SparkSession):
    # TODO: PARALLEL - check for correct handling of the situation when timer is execeeded
    # 1. after the first fold (parallelism=1)
    # 2. after several folds (parallelism > 1)
    pass
