#
# DATABRICKS CONFIDENTIAL & PROPRIETARY
# __________________
#
# Copyright 2020 Databricks, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property of Databricks, Inc.
# and its suppliers, if any.  The intellectual and technical concepts contained herein are
# proprietary to Databricks, Inc. and its suppliers and may be covered by U.S. and foreign Patents,
# patents in process, and are protected by trade secret and/or copyright law. Dissemination, use,
# or reproduction of this information is strictly forbidden unless prior written permission is
# obtained from Databricks, Inc.
#
# If you view or obtain a copy of this information and believe Databricks, Inc. may not have
# intended it to be made available, please promptly report it to Databricks Legal Department
# @ legal@databricks.com.
#

# pylint: disable=invalid-name
# pylint: disable=useless-super-delegation
# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-function-args
# pylint: disable=logging-format-interpolation
# pylint: disable=bare-except
# pylint: disable=ungrouped-imports
# pylint: disable=too-many-ancestors
# pylint: disable=too-many-branches
# pylint: disable=too-many-lines
import shutil
import tempfile
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasWeightCol, \
    HasPredictionCol, HasProbabilityCol, HasRawPredictionCol, HasValidationIndicatorCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import MLReadable, MLWritable
from pyspark.sql.functions import col, pandas_udf, countDistinct, rand
from xgboost import XGBClassifier, XGBRegressor
from xgboost.core import Booster
import cloudpickle
import xgboost
from xgboost.training import train as worker_train
from ..utils import get_logger, _get_max_num_concurrent_tasks
from .data import prepare_predict_data, prepare_train_val_data, convert_partition_data_to_dmatrix
from .model import (XgboostReader, XgboostWriter, XgboostModelReader,
                    XgboostModelWriter, deserialize_xgb_model,
                    serialize_xgb_model)
from .utils import (_get_default_params_from_func, get_class_name,
                    HasArbitraryParamsDict, HasBaseMarginCol, RabitContext,
                    _get_rabit_args, _get_args_from_message_list,
                    _get_spark_session)

try:
    from pyspark.databricks.sql.functions import unwrap_udt
    from pyspark.databricks.utils.instrumentation import instrumented
    from pyspark.ml.functions import array_to_vector
except:
    # For test. In CI we test against apache/spark and cannot install the
    # runtime-shim module as pyspark sub-module.
    from databricks.sql.functions import unwrap_udt, array_to_vector
    from databricks.utils.instrumentation import instrumented


def _get_cpu_per_task():
    return int(_get_spark_session().sparkContext.getConf().get('spark.task.cpus', '1'))


# Put pyspark specific params here, they won't be passed to XGBoost.
# like `validationIndicatorCol`, `baseMarginCol`
_pyspark_specific_params = [
    "featuresCol",
    "labelCol",
    "weightCol",
    "rawPredictionCol",
    "predictionCol",
    "probabilityCol",
    "validationIndicatorCol",
    "base_margin_col",
    "arbitraryParamsDict",
    "force_repartition",
    "num_workers",
    "use_gpu",
    "feature_names",
    "use_external_storage",
    "external_storage_precision",
]

_non_booster_params = [
    "missing",
    "n_estimators",
    "feature_weights",
]

_unsupported_xgb_params = [
    "gpu_id",  # we have "use_gpu" pyspark param instead.
    "enable_categorical",  # Use feature_types param to specify categorical feature instead
    "use_label_encoder",
    "feature_types",  # TODO
    "n_jobs",  # Do not allow user to set it, will use `spark.task.cpus` value instead.
    "nthread",  # Ditto
]

_unsupported_fit_params = {
    "sample_weight",  # Supported by spark param weightCol
    # Supported by spark param weightCol # and validationIndicatorCol
    "eval_set",
    "sample_weight_eval_set",
    "base_margin",  # Supported by spark param base_margin_col
}

_unsupported_predict_params = {
    # for classification, we can use rawPrediction as margin
    "output_margin",
    "validate_features",  # TODO
    "base_margin",  # Use pyspark base_margin_col param instead.
}


class _XgboostParams(HasFeaturesCol, HasLabelCol, HasWeightCol,
                     HasPredictionCol, HasValidationIndicatorCol,
                     HasArbitraryParamsDict, HasBaseMarginCol):

    num_workers = Param(
        Params._dummy(), "num_workers",
        "The number of XGBoost workers. Each XGBoost worker corresponds to one spark task.",
        TypeConverters.toInt)
    use_gpu = Param(
        Params._dummy(), "use_gpu",
        "A boolean variable. Set use_gpu=true if the executors " +
        "are running on GPU instances. Currently, only one GPU per task is supported."
    )
    force_repartition = Param(
        Params._dummy(), "force_repartition",
        "A boolean variable. Set force_repartition=true if you " +
        "want to force the input dataset to be repartitioned before XGBoost training." +
        "Note: The auto repartitioning judgement is not fully accurate, so it is recommended" +
        "to have force_repartition be True.")
    use_external_storage = Param(
        Params._dummy(), "use_external_storage",
        "A boolean variable (that is False by default). External storage is a parameter" +
        "for distributed training that allows external storage (disk) to be used when." +
        "you have an exceptionally large dataset. This should be set to false for" +
        "small datasets. Note that base margin and weighting doesn't work if this is True." +
        "Also note that you may use precision if you use external storage."
    )
    external_storage_precision = Param(
        Params._dummy(), "external_storage_precision",
        "The number of significant digits for data storage on disk when using external storage.",
        TypeConverters.toInt
    )
    feature_names = Param(
        Params._dummy(), "feature_names", "A list of str to specify feature names."
    )

    @classmethod
    def _xgb_cls(cls):
        """
        Subclasses should override this method and
        returns an xgboost.XGBModel subclass
        """
        raise NotImplementedError()

    def _get_xgb_model_creator(self):
        xgb_sklearn_params = self._gen_xgb_params_dict(gen_xgb_sklearn_estimator_param=True)
        # pylint: disable=unnecessary-lambda
        return lambda: self._xgb_cls()(**xgb_sklearn_params)

    # Parameters for xgboost.XGBModel()
    @classmethod
    def _get_xgb_params_default(cls):
        xgb_model_default = cls._xgb_cls()()
        params_dict = xgb_model_default.get_params()
        filtered_params_dict = {
            k: params_dict[k] for k in params_dict if k not in _unsupported_xgb_params
        }
        return filtered_params_dict

    def _set_xgb_params_default(self):
        filtered_params_dict = self._get_xgb_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_xgb_params_dict(self, gen_xgb_sklearn_estimator_param=False):
        xgb_params = {}
        non_xgb_params = (
            set(_pyspark_specific_params)
            | self._get_fit_params_default().keys()
            | self._get_predict_params_default().keys()
        )
        if not gen_xgb_sklearn_estimator_param:
            non_xgb_params |= set(_non_booster_params)
        for param in self.extractParamMap():
            if param.name not in non_xgb_params:
                xgb_params[param.name] = self.getOrDefault(param)

        arbitrary_params_dict = self.getOrDefault(self.arbitraryParamsDict)
        xgb_params.update(arbitrary_params_dict)
        return xgb_params

    # Parameters for xgboost.XGBModel().fit()
    @classmethod
    def _get_fit_params_default(cls):
        fit_params = _get_default_params_from_func(
            cls._xgb_cls().fit, _unsupported_fit_params
        )
        return fit_params

    def _set_fit_params_default(self):
        filtered_params_dict = self._get_fit_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_fit_params_dict(self):
        """
        Returns a dict of params for .fit()
        """
        fit_params_keys = self._get_fit_params_default().keys()
        fit_params = {}
        for param in self.extractParamMap():
            if param.name in fit_params_keys:
                fit_params[param.name] = self.getOrDefault(param)
        return fit_params

    # Parameters for xgboost.XGBModel().predict()
    @classmethod
    def _get_predict_params_default(cls):
        predict_params = _get_default_params_from_func(
            cls._xgb_cls().predict, _unsupported_predict_params)
        return predict_params

    def _set_predict_params_default(self):
        filtered_params_dict = self._get_predict_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_predict_params_dict(self):
        """
        Returns a dict of params for .predict()
        """
        predict_params_keys = self._get_predict_params_default().keys()
        predict_params = {}
        for param in self.extractParamMap():
            if param.name in predict_params_keys:
                predict_params[param.name] = self.getOrDefault(param)
        return predict_params

    def _validate_params(self):
        init_model = self.getOrDefault(self.xgb_model)
        if init_model is not None:
            if init_model is not None and not isinstance(init_model, Booster):
                raise ValueError(
                    'The xgb_model param must be set with a `xgboost.core.Booster` '
                    'instance.')

        missing_value = self.getOrDefault(self.missing)
        if missing_value != 0.0:
            # For the case missing value != 0 and sparse input case, the semantic in pyspark
            # is different with XGBoost library, so add warning here.
            get_logger(self.__class__.__name__).warning(
                f'We recommend using 0.0 as missing value to achieve better performance, '
                f'but you set missing param to be {missing_value}. In the case of missing != 0, '
                f'for features sparse vector input, the inactive values will be treated as 0 '
                f'instead of missing values, and the active values which are {missing_value} '
                f'will be treated as missing value, and this case the input sparse vector will '
                f'be densified when constructing XGBoost DMatrix, if feature sparsity is high '
                f'and input dataset is large, then it may slow down performance or lead to '
                f'out of memory.')

        if self.getOrDefault(self.num_workers) < 1:
            raise ValueError(
                f"Number of workers was {self.getOrDefault(self.num_workers)}."
                f"It cannot be less than 1 [Default is 1]")

        if self.getOrDefault(self.num_workers) > 1 and \
                self.isDefined(self.baseMarginCol) and self.getOrDefault(self.baseMarginCol):
            # TODO
            raise ValueError(
                "If `num_workers` > 1 or `use_external_storage` is True, setting `baseMarginCol` "
                "param is unsupported."
            )

        if self.getOrDefault(self.num_workers) == 1 and \
                self.getOrDefault(self.use_external_storage):
            raise ValueError(
                "If `num_workers` = 1, does not support setting `use_external_storage` to True."
            )

        if self.getOrDefault(self.use_gpu):
            tree_method = self.getParam("tree_method")
            if self.getOrDefault(
                    tree_method
            ) is not None and self.getOrDefault(tree_method) != "gpu_hist":
                raise ValueError(
                    f"tree_method should be 'gpu_hist' or None when use_gpu is True,"
                    f"found {self.getOrDefault(tree_method)}.")

            is_local = _is_local(_get_spark_session().sparkContext)

            if is_local:
                # Support GPU training in Spark local mode is just for debugging purposes,
                # so it's okay for printing the below warning instead of checking the real
                # gpu numbers and raising the exception.
                get_logger(self.__class__.__name__).warning(
                    "You enabled use_gpu in spark local mode. Please make sure your local node "
                    "has at least %d GPUs",
                    self.getOrDefault(self.num_workers),
                )
            else:
                gpu_per_task = _get_spark_session().sparkContext.getConf().get(
                    'spark.task.resource.gpu.amount')

                if not gpu_per_task or int(gpu_per_task) < 1:
                    raise RuntimeError(
                        "The spark cluster does not have the necessary GPU" +
                        "configuration for the spark task. Therefore, we cannot" +
                        "run xgboost training using GPU.")

                if int(gpu_per_task) > 1:
                    get_logger(self.__class__.__name__).warning(
                        f'You configured {gpu_per_task} GPU cores for each spark task, but in '
                        f'XGBoost training, every Spark task will only use one GPU core.'
                    )


def _is_local(spark_context) -> bool:
    """Whether it is Spark local mode"""
    # pylint: disable=protected-access
    return spark_context._jsc.sc().isLocal()


class _XgboostEstimator(Estimator, _XgboostParams, MLReadable, MLWritable):
    def __init__(self):
        super().__init__()
        self._set_xgb_params_default()
        self._set_fit_params_default()
        self._set_predict_params_default()
        # Note: The default value for arbitraryParamsDict must always be empty dict.
        #  For additional settings added into "arbitraryParamsDict" by default,
        #  they are added in `setParams`.
        self._setDefault(
            num_workers=1,
            use_gpu=False,
            force_repartition=False,
            feature_names=None,
            arbitraryParamsDict={},
            use_external_storage=False,
            external_storage_precision=5,  # Check if this needs to be modified
        )
        self._single_worker_nthread = _get_cpu_per_task()

    def setParams(self, **kwargs):
        _extra_params = {}
        if "arbitraryParamsDict" in kwargs:
            raise ValueError("Invalid param name: 'arbitraryParamsDict'.")

        if "nthread" in kwargs:
            if kwargs.get("num_workers", 1) > 1:
                raise ValueError(
                    "When setting `num_workers` > 1, you cannot specify `nthread` param."
                    "The `nthread` param is forced to apply `spark.task.cpus` config value."
                )

            kwargs = kwargs.copy()
            self._single_worker_nthread = kwargs.pop("nthread")
            get_logger(self.__class__.__name__).warning(
                "Setting `nthread` value for single worker training is not recommended,"
                "It might use CPU resources exceeding spark task resource limitation."
            )
        else:
            if kwargs.get("num_workers", 1) == 1:
                get_logger(self.__class__.__name__).warning(
                    f"The xgboost training will use single worker and set "
                    f"nthread={self._single_worker_nthread} (equal to `spark.task.cpus` config), "
                    f"If you need to increase threads number used in training, you can "
                    f"set `nthread` param."
                )

        for k, v in kwargs.items():
            if self.hasParam(k):
                self._set(**{str(k): v})
            else:
                if k in _unsupported_xgb_params or k in _unsupported_fit_params \
                        or k in _unsupported_predict_params:
                    raise ValueError(f"Unsupported param '{k}'.")
                _extra_params[k] = v
        _existing_extra_params = self.getOrDefault(self.arbitraryParamsDict)
        self._set(arbitraryParamsDict={**_existing_extra_params, **_extra_params})

    @classmethod
    def _pyspark_model_cls(cls):
        """
        Subclasses should override this method and
        returns a _XgboostModel subclass
        """
        raise NotImplementedError()

    def _create_pyspark_model(self, xgb_model):
        return self._pyspark_model_cls()(xgb_model)

    def _convert_to_sklearn_model(self, booster):
        xgb_sklearn_params = self._gen_xgb_params_dict(
            gen_xgb_sklearn_estimator_param=True
        )
        sklearn_model = self._xgb_cls()(**xgb_sklearn_params)
        sklearn_model._Booster = booster
        return sklearn_model

    def _query_plan_contains_valid_repartition(self, query_plan,
                                               num_partitions):
        """
        Returns true if the latest element in the logical plan is a valid repartition
        """
        start = query_plan.index("== Optimized Logical Plan ==")
        start += len("== Optimized Logical Plan ==") + 1
        num_workers = self.getOrDefault(self.num_workers)
        if query_plan[start:start +
                      len("Repartition"
                          )] == "Repartition" and num_workers == num_partitions:
            return True
        return False

    def _repartition_needed(self, dataset):
        """
        We repartition the dataset if the number of workers is not equal to the number of
        partitions. There is also a check to make sure there was "active partitioning"
        where either Round Robin or Hash partitioning was actively used before this stage.
        """
        if self.getOrDefault(self.force_repartition):
            return True
        try:
            num_partitions = dataset.rdd.getNumPartitions()
            query_plan = dataset._sc._jvm.PythonSQLUtils.explainString(
                dataset._jdf.queryExecution(), "extended")
            if self._query_plan_contains_valid_repartition(
                    query_plan, num_partitions):
                return False
        except:
            pass
        return True

    def _get_distributed_train_params(self, dataset):
        """
        This just gets the configuration params for distributed xgboost
        """
        params = self._gen_xgb_params_dict()
        fit_params = self._gen_fit_params_dict()
        verbose_eval = fit_params.pop("verbose", None)

        params.update(fit_params)
        params["verbose_eval"] = verbose_eval
        classification = self._xgb_cls() == XGBClassifier
        num_classes = int(dataset.select(countDistinct("label")).collect()[0][0])
        if classification and num_classes == 2:
            params["objective"] = "binary:logistic"
        elif classification and num_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_classes
        else:
            params["objective"] = "reg:squarederror"

        # TODO: support "num_parallel_tree" for random forest
        params["num_boost_round"] = self.getOrDefault(self.n_estimators)

        if self.getOrDefault(self.use_gpu):
            params["gpu_id"] = 0
            params["tree_method"] = "gpu_hist"

        return params

    @classmethod
    def _get_xgb_train_call_args(cls, train_params):
        xgb_train_default_args = _get_default_params_from_func(xgboost.train, {})
        booster_params, kwargs_params = {}, {}
        for key, value in train_params.items():
            if key in xgb_train_default_args:
                kwargs_params[key] = value
            else:
                booster_params[key] = value
        return booster_params, kwargs_params

    def _fit_distributed(self, xgb_model_creator, dataset, has_weight, has_validation):
        """
        Takes in the dataset, the other parameters, and produces a valid booster
        """
        num_workers = self.getOrDefault(self.num_workers)
        sc = _get_spark_session().sparkContext
        max_concurrent_tasks = _get_max_num_concurrent_tasks(sc)
        is_local = _is_local(sc)
        use_gpu = self.getOrDefault(self.use_gpu)

        if num_workers > max_concurrent_tasks:
            get_logger(self.__class__.__name__) \
                .warning(f'The num_workers {num_workers} set for xgboost distributed '
                         f'training is greater than current max number of concurrent '
                         f'spark task slots, you need wait until more task slots available '
                         f'or you need increase spark cluster workers.')

        has_validation = self.isDefined(self.validationIndicatorCol) and \
            self.getOrDefault(self.validationIndicatorCol)
        if self._repartition_needed(dataset) or has_validation:
            # If validationIndicatorCol defined, we always repartition dataset
            # to balance data, because user might unionise train and validation dataset,
            # without shuffling data then some partitions might contain only train or validation
            # dataset.
            # Repartition on `monotonically_increasing_id` column to avoid repartition
            # result unbalance. Directly using `.repartition(N)` might result in some
            # empty partitions.
            dataset = dataset.repartition(num_workers, rand(1))
        train_params = self._get_distributed_train_params(dataset)
        booster_params, train_call_kwargs_params = self._get_xgb_train_call_args(
            train_params
        )

        cpu_per_task = _get_cpu_per_task()
        dmatrix_kwargs = {
            "nthread": cpu_per_task,
            "feature_names": self.getOrDefault(self.feature_names),
            "feature_weights": self.getOrDefault(self.feature_weights),
            "missing": self.getOrDefault(self.missing),
        }
        booster_params["nthread"] = cpu_per_task

        def _train_booster_internal(pandas_df_iter, tmp_dir):
            """
            Takes in an RDD partition and outputs a booster for that partition after going through
            the Rabit Ring protocol
            """
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()

            if is_local and use_gpu:
                # If running on local mode spark cluster, set gpu_id to be partition id
                # instead of getting from task context gpu resouces.
                booster_params["gpu_id"] = context.partitionId()

            use_external_storage = self.getOrDefault(self.use_external_storage)
            external_storage_precision = self.getOrDefault(self.external_storage_precision)
            external_storage_path_prefix = tmp_dir
            dtrain, dval = None, []
            model = xgb_model_creator()

            context.barrier()
            _rabit_args = ""
            if context.partitionId() == 0:
                _rabit_args = str(_get_rabit_args(context, num_workers))

            messages = context.allGather(message=str(_rabit_args))
            _rabit_args = _get_args_from_message_list(messages)
            evals_result = {}
            with RabitContext(_rabit_args, context):
                if has_validation:
                    dtrain, dval = convert_partition_data_to_dmatrix(
                        pandas_df_iter, has_weight, model, has_validation,
                        use_external_storage, external_storage_path_prefix,
                        external_storage_precision,
                        dmatrix_kwargs=dmatrix_kwargs,
                    )
                    dval = [(dtrain, "training"), (dval, "validation")]
                else:
                    dtrain = convert_partition_data_to_dmatrix(
                        pandas_df_iter, has_weight, model, has_validation,
                        use_external_storage, external_storage_path_prefix,
                        external_storage_precision,
                        dmatrix_kwargs=dmatrix_kwargs,
                    )

                booster = worker_train(params=booster_params,
                                       dtrain=dtrain,
                                       evals=dval,
                                       evals_result=evals_result,
                                       **train_call_kwargs_params)
            context.barrier()

            if use_external_storage:
                shutil.rmtree(external_storage_path_prefix)
            if context.partitionId() == 0:
                yield pd.DataFrame(
                    data={'booster_bytes': [cloudpickle.dumps(booster)]})

        def _train_booster(pandas_df_iter):
            tmp_dir = tempfile.mkdtemp(prefix="sparkdl-xgboost-cache-")
            try:
                for x in _train_booster_internal(pandas_df_iter, tmp_dir):
                    yield x
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        result_ser_booster = dataset.mapInPandas(
            _train_booster,
            schema='booster_bytes binary').rdd.barrier().mapPartitions(
                lambda x: x).collect()[0][0]
        result_xgb_model = self._convert_to_sklearn_model(
            cloudpickle.loads(result_ser_booster))
        return self._copyValues(self._create_pyspark_model(result_xgb_model))

    @instrumented
    def _fit(self, dataset):
        self._validate_params()
        # Unwrap the VectorUDT type column "feature" to 4 primitive columns:
        # ['features.type', 'features.size', 'features.indices', 'features.values']
        features_col = col(self.getOrDefault(self.featuresCol))
        label_col = col(self.getOrDefault(self.labelCol)).alias('label')
        df_features_unwrapped_col = unwrap_udt(features_col).alias("features")
        select_cols = [df_features_unwrapped_col, label_col]

        has_weight = False
        has_validation = False
        has_base_margin = False

        if self.isDefined(self.weightCol) and self.getOrDefault(
                self.weightCol):
            has_weight = True
            select_cols.append(
                col(self.getOrDefault(self.weightCol)).alias('weight'))

        if self.isDefined(self.validationIndicatorCol) and \
                self.getOrDefault(self.validationIndicatorCol):
            has_validation = True
            select_cols.append(
                col(self.getOrDefault(
                    self.validationIndicatorCol)).alias('validationIndicator'))

        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True
            select_cols.append(
                col(self.getOrDefault(self.baseMarginCol)).alias("baseMargin"))

        dataset = dataset.select(*select_cols)
        flattened_columns = [
            'features.type', 'features.size', 'features.indices',
            'features.values'
        ] + dataset.columns[1:]
        dataset = dataset.select(*flattened_columns)
        # create local var `xgb_model_creator` to avoid pickle `self` object to remote worker
        xgb_model_creator = self._get_xgb_model_creator()  # pylint: disable=E1111
        fit_params = self._gen_fit_params_dict()

        if self.getOrDefault(self.num_workers) > 1:
            return self._fit_distributed(xgb_model_creator, dataset, has_weight, has_validation)

        # Note: fit_params will be pickled to remote, it may include `xgb_model` param
        #  which is used as initial model in training. The initial model will be a
        #  `Booster` instance which support pickling.

        _single_worker_nthread = self._single_worker_nthread

        use_gpu = self.getOrDefault(self.use_gpu)

        def train_func(pandas_df_iter):
            xgb_model = xgb_model_creator()
            xgb_model.set_params(n_jobs=_single_worker_nthread)

            if use_gpu:
                xgb_model.set_params(gpu_id=0, tree_method="gpu_hist")

            train_val_data = prepare_train_val_data(pandas_df_iter, has_weight,
                                                    xgb_model.missing,
                                                    has_validation,
                                                    has_base_margin)
            # We don't need to handle callbacks param in fit_params specially.
            # User need to ensure callbacks is pickle-able.
            if has_validation:
                train_X, train_y, train_w, train_base_margin, \
                     val_X, val_y, val_w, _ = train_val_data
                eval_set = [(val_X, val_y)]
                sample_weight_eval_set = [val_w]
                # base_margin_eval_set = [val_base_margin] <- the underline
                # Note that on XGBoost 1.2.0, the above doesn't exist.
                xgb_model.fit(train_X,
                              train_y,
                              sample_weight=train_w,
                              base_margin=train_base_margin,
                              eval_set=eval_set,
                              sample_weight_eval_set=sample_weight_eval_set,
                              **fit_params)
            else:
                train_X, train_y, train_w, train_base_margin = train_val_data
                xgb_model.fit(train_X,
                              train_y,
                              sample_weight=train_w,
                              base_margin=train_base_margin,
                              **fit_params)

            ser_model_string = serialize_xgb_model(xgb_model)
            yield pd.DataFrame(data={'model_string': [ser_model_string]})

        # Train on 1 remote worker, return the string of the serialized model
        result_ser_model_string = dataset.repartition(1) \
            .mapInPandas(train_func, schema='model_string string').collect()[0][0]

        # Load model
        result_xgb_model = deserialize_xgb_model(result_ser_model_string,
                                                 xgb_model_creator)
        return self._copyValues(self._create_pyspark_model(result_xgb_model))

    def write(self):
        return XgboostWriter(self)

    @classmethod
    def read(cls):
        return XgboostReader(cls)


class _XgboostModel(Model, _XgboostParams, MLReadable, MLWritable):
    def __init__(self, xgb_sklearn_model=None):
        super().__init__()
        self._xgb_sklearn_model = xgb_sklearn_model

    def get_booster(self):
        """
        Return the `xgboost.core.Booster` instance.
        """
        return self._xgb_sklearn_model.get_booster()

    def get_feature_importances(self, importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Parameters
        ----------
        importance_type: str, default 'weight'
            One of the importance types defined above.
        """
        return self.get_booster().get_score(importance_type=importance_type)

    def write(self):
        return XgboostModelWriter(self)

    @classmethod
    def read(cls):
        return XgboostModelReader(cls)

    def _transform(self, dataset):
        raise NotImplementedError()


class XgboostRegressorModel(_XgboostModel):
    """
    The model returned by :func:`sparkdl.xgboost.XgboostRegressor.fit`

    .. Note:: This API is experimental.
    """
    @classmethod
    def _xgb_cls(cls):
        return XGBRegressor

    def _transform(self, dataset):
        # Save missing, xgb_sklearn_model and predict_params to be local variable
        # to avoid the `self` object to be pickled to remote.
        missing = self._xgb_sklearn_model.missing
        xgb_sklearn_model = self._xgb_sklearn_model
        predict_params = self._gen_predict_params_dict()

        @pandas_udf('double')
        def predict_udf(iterator: Iterator[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]) \
                -> Iterator[pd.Series]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            for ftype, fsize, findices, fvalues in iterator:
                X, _, _, _ = prepare_predict_data(ftype, fsize, findices,
                                                  fvalues, missing, None)
                # Note: In every spark job task, pandas UDF will run in separate python process
                # so it is safe here to call the thread-unsafe model.predict method
                preds = xgb_sklearn_model.predict(X, validate_features=False, **predict_params)
                yield pd.Series(preds)

        @pandas_udf('double')
        def predict_udf_base_margin(iterator: Iterator[Tuple[pd.Series, pd.Series, \
             pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.Series]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            for ftype, fsize, findices, fvalues, base_margin in iterator:
                X, _, _, b_m = prepare_predict_data(ftype, fsize, findices,
                                                    fvalues, missing,
                                                    base_margin)
                # Note: In every spark job task, pandas UDF will run in separate python process
                # so it is safe here to call the thread-unsafe model.predict method
                preds = xgb_sklearn_model.predict(X,
                                                  base_margin=b_m,
                                                  validate_features=False,
                                                  **predict_params)
                yield pd.Series(preds)

        features_col = col(self.getOrDefault(self.featuresCol))
        features_col = unwrap_udt(features_col)

        has_base_margin = False
        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True

        if has_base_margin:
            base_margin_col = col(self.getOrDefault(self.baseMarginCol))
            pred_col = predict_udf_base_margin(features_col.type,
                                               features_col.size,
                                               features_col.indices,
                                               features_col.values,
                                               base_margin_col)
        else:
            pred_col = predict_udf(features_col.type, features_col.size,
                                   features_col.indices, features_col.values)

        predictionColName = self.getOrDefault(self.predictionCol)
        return dataset.withColumn(predictionColName, pred_col)


class XgboostClassifierModel(_XgboostModel, HasProbabilityCol,
                             HasRawPredictionCol):
    """
    The model returned by :func:`sparkdl.xgboost.XgboostClassifier.fit`

    .. Note:: This API is experimental.
    """
    @classmethod
    def _xgb_cls(cls):
        return XGBClassifier

    def _transform(self, dataset):
        # Save missing, xgb_sklearn_model and predict_params to be local variable
        # to avoid the `self` object to be pickled to remote.
        missing = self._xgb_sklearn_model.missing
        xgb_sklearn_model = self._xgb_sklearn_model
        predict_params = self._gen_predict_params_dict()

        @pandas_udf(
            'rawPrediction array<double>, prediction double, probability array<double>'
        )
        def predict_udf(iterator: Iterator[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]) \
                -> Iterator[pd.DataFrame]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            for ftype, fsize, findices, fvalues in iterator:
                X, _, _, _ = prepare_predict_data(ftype, fsize, findices,
                                                  fvalues, missing, None)
                # Note: In every spark job task, pandas UDF will run in separate python process
                # so it is safe here to call the thread-unsafe model.predict method
                margins = xgb_sklearn_model.predict(X,
                                                    output_margin=True,
                                                    validate_features=False,
                                                    **predict_params)
                if margins.ndim == 1:
                    # binomial case
                    classone_probs = expit(margins)
                    classzero_probs = 1.0 - classone_probs
                    raw_preds = np.vstack((-margins, margins)).transpose()
                    class_probs = np.vstack(
                        (classzero_probs, classone_probs)).transpose()
                else:
                    # multinomial case
                    raw_preds = margins
                    class_probs = softmax(raw_preds, axis=1)

                # It seems that they use argmax of class probs,
                # not of margin to get the prediction (Note: scala implementation)
                preds = np.argmax(class_probs, axis=1)
                yield pd.DataFrame(
                    data={
                        'rawPrediction': pd.Series(raw_preds.tolist()),
                        'prediction': pd.Series(preds),
                        'probability': pd.Series(class_probs.tolist())
                    })

        @pandas_udf(
            'rawPrediction array<double>, prediction double, probability array<double>'
        )
        def predict_udf_base_margin(iterator: Iterator[Tuple[pd.Series, pd.Series, \
            pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
            # deserialize model from ser_model_string, avoid pickling model to remote worker
            for ftype, fsize, findices, fvalues, base_margin in iterator:
                X, _, _, b_m = prepare_predict_data(ftype, fsize, findices,
                                                    fvalues, missing,
                                                    base_margin)
                # Note: In every spark job task, pandas UDF will run in separate python process
                # so it is safe here to call the thread-unsafe model.predict method
                margins = xgb_sklearn_model.predict(X,
                                                    base_margin=b_m,
                                                    output_margin=True,
                                                    validate_features=False,
                                                    **predict_params)
                if margins.ndim == 1:
                    # binomial case
                    classone_probs = expit(margins)
                    classzero_probs = 1.0 - classone_probs
                    raw_preds = np.vstack((-margins, margins)).transpose()
                    class_probs = np.vstack(
                        (classzero_probs, classone_probs)).transpose()
                else:
                    # multinomial case
                    raw_preds = margins
                    class_probs = softmax(raw_preds, axis=1)

                # It seems that they use argmax of class probs,
                # not of margin to get the prediction (Note: scala implementation)
                preds = np.argmax(class_probs, axis=1)
                yield pd.DataFrame(
                    data={
                        'rawPrediction': pd.Series(raw_preds.tolist()),
                        'prediction': pd.Series(preds),
                        'probability': pd.Series(class_probs.tolist())
                    })

        features_col = col(self.getOrDefault(self.featuresCol))
        features_col = unwrap_udt(features_col)

        has_base_margin = False
        if self.isDefined(self.baseMarginCol) and self.getOrDefault(
                self.baseMarginCol):
            has_base_margin = True

        if has_base_margin:
            base_margin_col = col(self.getOrDefault(self.baseMarginCol))
            pred_struct = predict_udf_base_margin(features_col.type,
                                                  features_col.size,
                                                  features_col.indices,
                                                  features_col.values,
                                                  base_margin_col)
        else:
            pred_struct = predict_udf(features_col.type, features_col.size,
                                      features_col.indices,
                                      features_col.values)

        pred_struct_col = '_prediction_struct'

        rawPredictionColName = self.getOrDefault(self.rawPredictionCol)
        predictionColName = self.getOrDefault(self.predictionCol)
        probabilityColName = self.getOrDefault(self.probabilityCol)
        dataset = dataset.withColumn(pred_struct_col, pred_struct)
        if rawPredictionColName:
            dataset = dataset.withColumn(
                rawPredictionColName,
                array_to_vector(col(pred_struct_col).rawPrediction))
        if predictionColName:
            dataset = dataset.withColumn(predictionColName,
                                         col(pred_struct_col).prediction)
        if probabilityColName:
            dataset = dataset.withColumn(
                probabilityColName,
                array_to_vector(col(pred_struct_col).probability))

        return dataset.drop(pred_struct_col)


def _set_pyspark_xgb_cls_param_attrs(pyspark_estimator_class,
                                     pyspark_model_class):
    params_dict = pyspark_estimator_class._get_xgb_params_default()

    def param_value_converter(v):
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        elif isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        elif isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        else:
            return v

    def set_param_attrs(attr_name, param_obj_):
        param_obj_.typeConverter = param_value_converter
        setattr(pyspark_estimator_class, attr_name, param_obj_)
        setattr(pyspark_model_class, attr_name, param_obj_)

    for name in params_dict.keys():
        if name == 'missing':
            doc = 'Specify the missing value in the features, default np.nan. ' \
                  'We recommend using 0.0 as the missing value for better performance. ' \
                  'Note: In a spark DataFrame, the inactive values in a sparse vector ' \
                  'mean 0 instead of missing values, unless missing=0 is specified.'
        else:
            doc = f'Refer to XGBoost doc of ' \
                  f'{get_class_name(pyspark_estimator_class._xgb_cls())} for this param {name}'

        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    fit_params_dict = pyspark_estimator_class._get_fit_params_default()
    for name in fit_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}' \
              f'.fit() for this param {name}'
        if name == 'callbacks':
            doc += 'The callbacks can be arbitrary functions. It is saved using cloudpickle ' \
                   'which is not a fully self-contained format. It may fail to load with ' \
                   'different versions of dependencies.'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

    predict_params_dict = pyspark_estimator_class._get_predict_params_default()
    for name in predict_params_dict.keys():
        doc = f'Refer to XGBoost doc of {get_class_name(pyspark_estimator_class._xgb_cls())}' \
              f'.predict() for this param {name}'
        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)
