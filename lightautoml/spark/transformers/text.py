from copy import copy
from typing import List, Optional, Dict

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.sql import functions as F

from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.dataset import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.text import TunableTransformer, text_check

import numpy as np

class TfidfTextTransformer(SparkTransformer, TunableTransformer):
    """Simple Tfidf vectorizer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidf"
    _default_params = {
        "min_df": 5,
        "max_df": 1.0,
        "max_features": 30_000,
        "ngram_range": (1, 1),
        "analyzer": "word",
        "dtype": np.float32,
    }

    @property
    def features(self) -> List[str]:
        """Features list."""

        return self._features

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        subs: Optional[float] = None,
        random_state: int = 42,
    ):
        """

        Args:
            default_params: algo hyperparams.
            freeze_defaults: Flag.
            subs: Subsample to calculate freqs. If ``None`` - full data.
            random_state: Random state to take subsample.

        Note:
            The behaviour of `freeze_defaults`:

            - ``True`` :  params may be rewritten depending on dataset.
            - ``False``:  params may be changed only
              manually or with tuning.

        """
        super().__init__(default_params, freeze_defaults)
        self.subs = subs
        self.random_state = random_state
        self.idf_columns_pipelines: Optional[Dict[Pipeline]] = None

    def init_params_on_input(self, dataset: SparkDataset) -> dict:
        """Get transformer parameters depending on dataset parameters.

        Args:
            dataset: Dataset used for model parmaeters initialization.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        suggested_params = copy(self.default_params)
        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        # TODO: decide later what to do with this part
        # rows_num = len(dataset.data)
        # if rows_num > 50_000:
        #     suggested_params["min_df"] = 25

        return suggested_params

    def fit(self, dataset: SparkDataset):
        """Fit tfidf vectorizer.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        if self._params is None:
            self.params = self.init_params_on_input(dataset)

        sdf = dataset.data
        if self.subs:
            sdf = sdf.sample(self.subs, seed=self.random_state)
        sdf = sdf.fillna("")

        self.idf_columns_pipelines = dict()
        feats = []
        for c in sdf.columns:
            # TODO: set params here from self.params
            tokenizer = Tokenizer(inputCol=c, outputCol=f"{c}_words")
            count_tf = CountVectorizer(inputCol=tokenizer.getOutputCol(), outputCol=f"{c}_word_features")
            idf = IDF(inputCol=count_tf.getOutputCol(), outputCol=f"{c}_idf_features")
            pipeline = Pipeline(stages=[tokenizer, count_tf, idf])

            tfidf_pipeline_model = pipeline.fit(sdf)

            features = list(
                np.char.array([self._fname_prefix + "_"])
                + np.arange(count_tf.getVocabSize()).astype(str)
                + np.char.array(["__" + c])
            )
            feats.extend(features)

            self.idf_columns_pipelines[c] = (tfidf_pipeline_model, features)

        self._features = feats

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Sparse dataset with encoded text.

        """
        # checks here
        super().transform(dataset)

        sdf = dataset.data
        sdf = sdf.fillna("")

        # transform
        roles = NumericRole()
        curr_sdf = sdf
        for c in sdf.columns:
            tfidf_model, _ = self.idf_columns_pipelines[c]
            curr_sdf = tfidf_model.transform(curr_sdf)

        new_sdf = curr_sdf.select(self._features)

        output = dataset.empty()
        output.set_data(new_sdf, self._features, roles)

        return output
