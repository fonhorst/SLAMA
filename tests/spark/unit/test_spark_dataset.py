import os
import shutil

from lightautoml.dataset.roles import NumericRole
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from . import spark as spark_sess
import numpy as np

spark = spark_sess


def test_spark_dataset_save_load(spark: SparkSession):
    path = "/tmp/test_slama_ds.dataset"

    if os.path.exists(path):
        shutil.rmtree(path)

    df = spark.createDataFrame([{
        SparkDataset.ID_COLUMN: i,
        "a": i + 1,
        "b": i * 10 + 1
    } for i in range(10)])

    ds = SparkDataset(data=df, roles={"a": NumericRole(dtype=np.int32), "b": NumericRole(dtype=np.int32)})

    ds.save(path=path)

    loaded_ds = SparkDataset.load(path=path)

    assert loaded_ds.uid
    assert loaded_ds.uid != ds.uid
    assert loaded_ds.name == ds.name
    assert loaded_ds.roles == ds.roles
    assert loaded_ds.data.schema == ds.data.schema

    if os.path.exists(path):
        shutil.rmtree(path)
