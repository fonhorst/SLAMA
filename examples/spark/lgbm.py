import math
from typing import List

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

    df = ds.data.cache()
    df.write.format("noop").mode("overwrite").save()

    train, test = ds.data.randomSplit([0.85, 0.15], seed=42)
    # train = ds.data
    # test = train

    feats = list(sorted(ds.features))

    def splits(features: List[str], count: int):
        elts_per_split = math.ceil(len(features) / count)
        for i in range(count):
            yield features[i * elts_per_split: (i + 1) * elts_per_split]

    # feats = ['FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6']


    # good
    # feats = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']
    # good
    # feats = ['DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18']
    # bad
    feats = ['FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE']

    for _ in range(1):
        for i, feature_cols in enumerate(splits(feats, count=1)):
            print(f"Id: {i} NUM FEATS: {len(feature_cols)} FEATURE COLS: {feature_cols}")

            train.select(feature_cols).printSchema()

            featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
            train_data = featurizer.transform(train)[ds.target_column, "features"]
            test_data = featurizer.transform(test)[ds.target_column, "features"]

            print("===============================")

            train_data.printSchema()

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
    attempts_count = 1
    failed_attempts = 0
    for attempt_id in range(attempts_count):
        failed = False
        try:
            main()
        except:
            failed = True
            failed_attempts += 1

    print(f"STATISTICS: {(attempts_count - failed_attempts)}/{attempts_count} successful attempts")
