from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset


def test_ml_pipeline(spark: SparkSession, dataset: SparkDataset):
    # acc
    # DummyMlAlgo
    SparkMLPipeline()
    pass