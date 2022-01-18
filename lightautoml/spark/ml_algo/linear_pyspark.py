"""Linear models for tabular datasets."""

import logging
from copy import copy
from typing import Tuple, Optional
from typing import Union

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

from .base import TabularMLAlgo, SparkMLModel
from ..dataset.base import SparkDataset, SparkDataFrame
from ...utils.timer import TaskTimer

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class LinearLBFGS(TabularMLAlgo):

    _name: str = "LinearL2"

    def __init__(self, timer: Optional[TaskTimer] = None, **params):
        super().__init__()

        self._prediction_col = f"prediction_{self._name}"
        self.params = params
        self.task = None
        self._timer = timer

    def _infer_params(self, train: SparkDataset) -> Pipeline:
        logger.debug("Building pipeline in linear lGBFS")
        params = copy(self.params)

        # categorical features
        cat_feats = [feat for feat in train.features if train.roles[feat].name == "Category"]
        non_cat_feats = [feat for feat in train.features if train.roles[feat].name != "Category"]

        ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_{self._name}_ohe" for f in cat_feats])
        assembler = VectorAssembler(
            inputCols=non_cat_feats + ohe.getOutputCols(),
            outputCol=f"{self._name}_vassembler_features"
        )

        # TODO: SPARK-LAMA add params processing later
        if self.task.name in ["binary", "multiclass"]:
            model = LogisticRegression(featuresCol=assembler.getOutputCol(),
                                       labelCol=train.target_column,
                                       predictionCol=self._prediction_col)
                                       # **params)
        elif self.task.name == "reg":
            model = LinearRegression(featuresCol=assembler.getOutputCol(),
                                     labelCol=train.target_column,
                                     predictionCol=self._prediction_col)
                                     # **params)
            model.setSolver("l-bfgs")
        else:
            raise ValueError("Task not supported")

        pipeline = Pipeline(stages=[ohe, assembler, model])

        logger.debug("The pipeline is completed in linear lGBFS")
        return pipeline

    def fit_predict_single_fold(
        self, train: SparkDataset, valid: SparkDataset
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        logger.info(f"predict single fold in LinearLBGFS")
        if self.task is None:
            self.task = train.task

        # TODO: SPARK-LAMA target column?
        train_sdf = self._make_sdf_with_target(train)
        val_sdf = valid.data

        pipeline = self._infer_params(train)
        ml_model = pipeline.fit(train_sdf)

        val_pred = ml_model.transform(val_sdf)

        return ml_model, val_pred, self._prediction_col

    def predict_single_fold(self, dataset: SparkDataset, model: SparkMLModel) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.transform(dataset.data)
        return pred