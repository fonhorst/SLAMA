from examples_utils import get_spark_session
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.feature import VectorAssembler
from synapse.ml.train import ComputeModelStatistics


def main():
    spark = get_spark_session(4)

    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(
            "file:///opt/spark_data/company_bancruptacy_prediction.csv"
        )
    )

    train, test = df.randomSplit([0.85, 0.15], seed=1)

    feature_cols = df.columns[1:]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["Bankrupt?", "features"]
    test_data = featurizer.transform(test)["Bankrupt?", "features"]

    model = LightGBMClassifier(
        objective="binary",
        featuresCol="features",
        labelCol="Bankrupt?",
        isUnbalance=True,
        executionMode="streaming",
        isProvideTrainingMetric=True,
        verbosity=1
    )

    model = model.fit(train_data)

    predictions = model.transform(test_data)

    metrics = ComputeModelStatistics(
        evaluationMetric="classification",
        labelCol="Bankrupt?",
        scoredLabelsCol="prediction",
    ).transform(predictions)

    metrics_df = metrics.toPandas()

    print(metrics_df.head(10))

    spark.stop()


if __name__ == "__main__":
    main()
