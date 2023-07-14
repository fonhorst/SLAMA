from examples_utils import get_spark_session
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.feature import VectorAssembler
from synapse.ml.train import ComputeModelStatistics

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from pyspark.sql import functions as sf


def main():
    spark = get_spark_session(4)

    ds = SparkDataset.load(
        path=f"/opt/spark_data/preproccessed_datasets/lama_test_dataset__lgb_adv__features.dataset",
        persistence_manager=PlainCachePersistenceManager(),
        partitions_num=4
    )

    row = ds.data.select(*[sf.sum(sf.isnan(f).astype("int")).alias(f) for f in ds.features]).first()
    nan_feats = {feat: count for feat, count in row.asDict().items() if count > 0}
    good_feats = list(set(ds.features).difference(set(nan_feats.keys())))
    ds = ds[:, good_feats]

    train, test = ds.data.randomSplit([0.85, 0.15], seed=1)

    feature_cols = ds.features
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="error")
    train_data = featurizer.transform(train)[ds.target_column, "features"]
    test_data = featurizer.transform(test)[ds.target_column, "features"]

    # df = (
    #     spark.read.format("csv")
    #         .option("header", True)
    #         .option("inferSchema", True)
    #         .load(
    #         "file:///opt/spark_data/company_bancruptacy_prediction.csv"
    #     )
    # )
    #
    # train, test = df.randomSplit([0.85, 0.15], seed=1)
    #
    # feature_cols = df.columns[1:]
    # featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    # train_data = featurizer.transform(train)["Bankrupt?", "features"]
    # test_data = featurizer.transform(test)["Bankrupt?", "features"]

    model = LightGBMClassifier(
        objective="binary",
        featuresCol="features",
        labelCol=ds.target_column,
        isUnbalance=True,
        executionMode="streaming",
        isProvideTrainingMetric=True,
        verbosity=1
    )

    model = model.fit(train_data)

    predictions = model.transform(test_data)

    metrics = ComputeModelStatistics(
        evaluationMetric="classification",
        labelCol=ds.target_column,
        scoredLabelsCol="prediction",
    ).transform(predictions)

    metrics_df = metrics.toPandas()

    print(metrics_df.head(10))

    spark.stop()


if __name__ == "__main__":
    main()
