from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset


def test_ml_pipeline(spark: SparkSession, dataset: SparkDataset):
    # acc
    # DummyMlAlgo
    iterator = SparkFoldsIterator(dataset)

    ml_pipeline = SparkMLPipeline(
        ml_algos=[
            SparkBoostLGBM(
                default_params={'numIterations': 25},
                use_single_dataset_mode=True,
                use_barrier_execution_mode=True
            ),
            SparkLinearLBFGS(
                default_params={'regParam': [1e-5]}
            )
        ]
    )

    oof_preds = ml_pipeline.fit_predict(iterator)
    test_preds = ml_pipeline.predict(dataset)

    for pred_feat in ml_pipeline.output_roles.keys():
        assert pred_feat in oof_preds.features
        assert pred_feat in oof_preds.data.columns
        assert pred_feat in test_preds.features
        assert pred_feat in test_preds.data.columns

        assert oof_preds.data.count() == count
        assert test_preds.data.count() == dataset.data.count()

        score = dataset.task.get_dataset_metric()
        oof_metric = score(oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            sf.col(dataset.target_column).alias('target'),
            sf.col(ml_algo.prediction_feature).alias('prediction')
        ))
        test_metric = score(test_preds.data.select(
            SparkDataset.ID_COLUMN,
            sf.col(dataset.target_column).alias('target'),
            sf.col(ml_algo.prediction_feature).alias('prediction')
        ))

        assert oof_metric > 0
        assert test_metric > 0

    pass
