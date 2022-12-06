import codecs
import os

import yaml

from mlflow.exceptions import MissingConfigException
from mlflow.utils.file_utils import ENCODING, exists


def write_yaml(root, file_name, data, overwrite=False, sort_keys=True):
    """
    Write dictionary data in yaml format.

    :param root: Directory name.
    :param file_name: Desired file name. Will automatically add .yaml extension if not given
    :param data: data to be dumped as yaml format
    :param overwrite: If True, will overwrite existing files
    """
    if not exists(root):
        raise MissingConfigException("Parent directory '%s' does not exist." % root)

    file_path = os.path.join(root, file_name)
    yaml_file_name = file_path if file_path.endswith(".yaml") else file_path + ".yaml"

    if exists(yaml_file_name) and not overwrite:
        raise Exception("Yaml file '%s' exists as '%s" % (file_path, yaml_file_name))

    try:
        with codecs.open(yaml_file_name, mode="w", encoding=ENCODING) as yaml_file:
            yaml.safe_dump(
                data,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=sort_keys,
            )
    except Exception as e:
        raise e
