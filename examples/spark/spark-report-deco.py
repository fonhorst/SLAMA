import logging.config

from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.report import SparkReportDeco
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_timer
from sparklightautoml.utils import logging_config

logging.config.dictConfig(logging_config(log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 5
    use_algos = [["lgb"]]
    dataset_name = "lama_test_dataset"
    dataset = get_dataset(dataset_name)

    train_data, test_data = prepare_test_and_train(dataset, seed)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(dataset.task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 3000}},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False, 'random_state': seed}
        )

        report_automl = SparkReportDeco(
            output_path="/tmp/spark",
            report_file_name="spark_lama_report.html",
            interpretation=True
        )(automl)

        oof_preds = report_automl.fit_predict(
            train_data,
            roles=dataset.roles,
            persistence_manager=get_persistence_manager()
        ).persist()
        report_automl.predict(test_data, add_reader_attrs=True)
        oof_preds.unpersist()
        # this is necessary if persistence_manager is of CompositeManager type
        automl.persistence_manager.unpersist_all()
