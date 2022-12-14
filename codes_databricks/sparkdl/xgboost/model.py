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
# pylint: disable=logging-format-interpolation
# pylint: disable=unnecessary-lambda

import base64
import os
import uuid

from pyspark import cloudpickle
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml.util import (DefaultParamsReader, DefaultParamsWriter,
                             MLReader, MLWriter)
from xgboost.core import Booster

from sparkdl.utils import get_logger
from .utils import get_class_name


def _get_or_create_tmp_dir():
    root_dir = SparkFiles.getRootDirectory()
    xgb_tmp_dir = os.path.join(root_dir, 'xgboost-tmp')
    if not os.path.exists(xgb_tmp_dir):
        os.makedirs(xgb_tmp_dir)
    return xgb_tmp_dir


def serialize_xgb_model(model):
    """
    Serialize the input model to a string.
    :param model: an xgboost.XGBModel instance,
      such as xgboost.XGBClassifier or xgboost.XGBRegressor instance
    """
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f'{uuid.uuid4()}.json')
    model.save_model(tmp_file_name)
    with open(tmp_file_name) as f:
        ser_model_string = f.read()
    return ser_model_string


def deserialize_xgb_model(ser_model_string, xgb_model_creator):
    """
    Deserialize an xgboost.XGBModel instance from the input ser_model_string.
    """
    xgb_model = xgb_model_creator()
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f'{uuid.uuid4()}.json')
    with open(tmp_file_name, "w") as f:
        f.write(ser_model_string)
    xgb_model.load_model(tmp_file_name)
    return xgb_model


def serialize_booster(booster):
    """
    Serialize the input booster to a string.
    :param booster: an xgboost.core.Booster instance
    """
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f'{uuid.uuid4()}.json')
    booster.save_model(tmp_file_name)
    with open(tmp_file_name) as f:
        ser_model_string = f.read()
    return ser_model_string


def deserialize_booster(ser_model_string):
    """
    Deserialize an xgboost.core.Booster from the input ser_model_string.
    """
    booster = Booster()
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f'{uuid.uuid4()}.json')
    with open(tmp_file_name, "w") as f:
        f.write(ser_model_string)
    booster.load_model(tmp_file_name)
    return booster


_INIT_BOOSTER_SAVE_PATH = "init_booster.json"


def _get_spark_session():
    return SparkSession.builder.getOrCreate()


class XgboostSharedReadWrite:

    @staticmethod
    def saveMetadata(instance, path, sc, logger, extraMetadata=None):
        """
        Save the metadata of an sparkdl.xgboost._XgboostEstimator or
        sparkdl.xgboost._XgboostModel.
        """
        instance._validate_params()
        skipParams = ['callbacks', 'xgb_model']
        jsonParams = {}
        for p, v in instance._paramMap.items():
            if p.name not in skipParams:
                jsonParams[p.name] = v

        extraMetadata = extraMetadata or {}
        callbacks = instance.getOrDefault(instance.callbacks)
        if callbacks is not None:
            logger.warning('The callbacks parameter is saved using cloudpickle and it '
                           'is not a fully self-contained format. It may fail to load '
                           'with different versions of dependencies.')
            serialized_callbacks = \
                base64.encodebytes(cloudpickle.dumps(callbacks)).decode('ascii')
            extraMetadata['serialized_callbacks'] = serialized_callbacks
        init_booster = instance.getOrDefault(instance.xgb_model)
        if init_booster is not None:
            extraMetadata['init_booster'] = _INIT_BOOSTER_SAVE_PATH
        DefaultParamsWriter.saveMetadata(
            instance, path, sc, extraMetadata=extraMetadata, paramMap=jsonParams)
        if init_booster is not None:
            ser_init_booster = serialize_booster(init_booster)
            save_path = os.path.join(path, _INIT_BOOSTER_SAVE_PATH)
            _get_spark_session().createDataFrame(
                [(ser_init_booster,)], ['init_booster']).write.parquet(save_path)

    @staticmethod
    def loadMetadataAndInstance(pyspark_xgb_cls, path, sc, logger):
        """
        Load the metadata and the instance of an sparkdl.xgboost._XgboostEstimator or
        sparkdl.xgboost._XgboostModel.

        :return: a tuple of (metadata, instance)
        """
        metadata = DefaultParamsReader.loadMetadata(
            path, sc, expectedClassName=get_class_name(pyspark_xgb_cls))
        pyspark_xgb = pyspark_xgb_cls()
        DefaultParamsReader.getAndSetParams(pyspark_xgb, metadata)

        if 'serialized_callbacks' in metadata:
            serialized_callbacks = metadata['serialized_callbacks']
            try:
                callbacks = \
                    cloudpickle.loads(base64.decodebytes(serialized_callbacks.encode('ascii')))
                pyspark_xgb.set(pyspark_xgb.callbacks, callbacks)
            except Exception as e:   # pylint: disable=W0703
                logger.warning('Fails to load the callbacks param due to {}. Please set the '
                               'callbacks param manually for the loaded estimator.'.format(e))

        if 'init_booster' in metadata:
            load_path = os.path.join(path, metadata['init_booster'])
            ser_init_booster = _get_spark_session().read.parquet(load_path) \
                .collect()[0].init_booster
            init_booster = deserialize_booster(ser_init_booster)
            pyspark_xgb.set(pyspark_xgb.xgb_model, init_booster)

        pyspark_xgb._resetUid(metadata["uid"])
        return metadata, pyspark_xgb


class XgboostWriter(MLWriter):

    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level='WARN')

    def saveImpl(self, path):
        XgboostSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)


class XgboostReader(MLReader):

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level='WARN')

    def load(self, path):
        _, pyspark_xgb = XgboostSharedReadWrite \
            .loadMetadataAndInstance(self.cls, path, self.sc, self.logger)
        return pyspark_xgb


class XgboostModelWriter(MLWriter):

    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level='WARN')

    def saveImpl(self, path):
        """
        Save metadata and model for a :py:class:`_XgboostModel`
        - save metadata to path/metadata
        - save model to path/model.json
        """
        xgb_model = self.instance._xgb_sklearn_model
        XgboostSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)
        model_save_path = os.path.join(path, "model.json")
        ser_xgb_model = serialize_xgb_model(xgb_model)
        _get_spark_session().createDataFrame(
            [(ser_xgb_model,)], ['xgb_sklearn_model']).write.parquet(model_save_path)


class XgboostModelReader(MLReader):

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level='WARN')

    def load(self, path):
        """
        Load metadata and model for a :py:class:`_XgboostModel`

        :return: XgboostRegressorModel or XgboostClassifierModel instance
        """
        _, py_model = XgboostSharedReadWrite.loadMetadataAndInstance(
            self.cls, path, self.sc, self.logger)

        xgb_sklearn_params = py_model._gen_xgb_params_dict(gen_xgb_sklearn_estimator_param=True)
        model_load_path = os.path.join(path, "model.json")

        ser_xgb_model = _get_spark_session().read.parquet(model_load_path) \
            .collect()[0].xgb_sklearn_model
        xgb_model = deserialize_xgb_model(
            ser_xgb_model,
            lambda: self.cls._xgb_cls()(**xgb_sklearn_params)
        )
        py_model._xgb_sklearn_model = xgb_model
        return py_model
