import numpy as np
import pandas as pd
import pytest
import torch
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import PathRole, NumericRole
from lightautoml.image.utils import pil_loader
from lightautoml.spark.transformers.image import PathBasedAutoCVWrap as SparkPathBasedAutoCVWrap, \
    ImageFeaturesTransformer as SparkImageFeaturesTransformer
from lightautoml.transformers.text import AutoNLPWrap
from lightautoml.spark.transformers.text import AutoNLPWrap as SparkAutoNLPWrap
from lightautoml.transformers.image import ImageFeaturesTransformer
from . import compare_by_content
from .test_transformers import smoke_check


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


@pytest.fixture
def image_dataset() -> PandasDataset:
    source_data = pd.DataFrame(data={
        "path_a": [f"resources/images/cat_{i + 1}.jpg" for i in range(3)],
        "path_b": [f"resources/images/cat_{i + 1}.jpg" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    return ds


@pytest.fixture
def text_dataset() -> PandasDataset:
    source_data = pd.DataFrame(data={
        "text_a": [f"Lorem Ipsum is simply dummy text of the printing and typesetting industry. {i}" for i in range(3)],
        "text_b": [f"Lorem Ipsum is simply dummy text of the printing and typesetting industry. {i}" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    return ds


def test_path_auto_cv_wrap(spark: SparkSession, image_dataset: PandasDataset):
    result_ds = smoke_check(spark, image_dataset, SparkPathBasedAutoCVWrap(image_loader=pil_loader,
                                                                           device=torch.device("cpu:0")))

    # TODO: replace with a normal content check
    assert result_ds.shape[0] == image_dataset.shape[0]
    assert all(isinstance(role, NumericRole) and role.dtype == np.float32 for c, role in result_ds.roles.items())


def test_image_features_transformer(spark: SparkSession, image_dataset: PandasDataset):
    compare_by_content(spark, image_dataset,
                       ImageFeaturesTransformer(n_jobs=1, loader=pil_loader),
                       SparkImageFeaturesTransformer(n_jobs=1, loader=pil_loader))


@pytest.mark.skip
def test_auto_nlp_wrap(spark: SparkSession, text_dataset: PandasDataset):
    kwargs = {
        "model_name": "random_lstm",
        "device": torch.device("cpu:0"),
        "embedding_model": None
    }
    compare_by_content(spark, text_dataset,
                       AutoNLPWrap(**kwargs),
                       SparkAutoNLPWrap(**kwargs))
