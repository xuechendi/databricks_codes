from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.constants import MERGE
from pyspark.sql import DataFrame
from pyspark.sql.streaming import StreamingQuery
from functools import wraps
from typing import List, Dict, Union, Any


def feature_table(func):  # noqa: D212,D400,D415
    """
    .. note::

       Experimental: This decorator may change or be
       removed in a future release without warning.

    The ``@feature_table`` decorator specifies that a
    function is used to generate feature data.
    Functions decorated with ``@feature_table`` must return a
    single :class:`DataFrame <pyspark.sql.DataFrame>`, which will be
    written to Feature Store. For example:

    .. code-block:: python

       from databricks.feature_store import feature_table

       @feature_table
       def compute_customer_features(data):
           '''Feature computation function that takes raw
              data and returns a DataFrame of features.'''
           return (data.groupBy('cid')
             .agg(count('*').alias('num_purchases'))
           )

    A function that is decorated with the ``@feature_table`` decorator will gain these function attributes:

    .. function:: compute_and_write(input: Dict[str, Any], feature_table_name: str, mode: str = 'merge') -> pyspark.sql.dataframe.DataFrame

       .. note::

          Experimental: This function may change or be removed in
          a future release without warning.

       Calls the decorated function using the provided ``input``, then writes
       the output :class:`DataFrame <pyspark.sql.DataFrame>` to the feature table specified by ``feature_table_name``.

       .. code-block:: python
          :caption: Example:

          compute_customer_features.compute_and_write(
              input={
                  'data': data,
              },
              feature_table_name='recommender_system.customer_features',
              mode='merge'
          )

       :param input: If ``input`` is not a dictionary,
          it is passed to the decorated function as the first positional argument.
          If ``input`` is a dictionary, the contents are unpacked and passed to
          the decorated function as keyword arguments.
       :param feature_table_name: A feature table name of the form
          ``<database_name>.<table_name>``, for example ``dev.user_features``.
          Raises exception if this feature table does not exist.
       :param mode: Two supported write modes: ``"overwrite"`` updates the whole table,
          while ``"merge"`` will upsert the rows in ``df`` into the feature table.
       :return: :class:`DataFrame <pyspark.sql.DataFrame>` (``df``) containing feature values.

    .. function:: compute_and_write_streaming(input: Dict[str, Any], feature_table_name: str, checkpoint_location: Optional[str] = None, trigger: Dict[str, Any] = {'processingTime': '5 minutes'}) -> pyspark.sql.streaming.StreamingQuery

       .. note::

          Experimental: This function may change or be removed in
          a future release without warning.

       Calls the decorated function using the provided input,
       then streams the output :class:`DataFrame <pyspark.sql.DataFrame>` to the
       feature table specified by ``feature_table_name``.

       .. code-block:: python
          :caption: Example:

          compute_customer_features.compute_and_write_streaming(
              input={
                  'data': data,
              },
              feature_table_name='recommender_system.customer_features',
          )

       :param input: If ``input`` is not a dictionary,
          it is passed to the decorated function as the first positional argument.
          If ``input`` is a dictionary, the contents are unpacked and passed
          to the decorated function as keyword arguments.
       :param feature_table_name: A feature table name of the form ``<database_name>.<table_name>``,
          for example ``dev.user_features``.
       :param checkpoint_location: Sets the Structured Streaming ``checkpointLocation`` option.
          By setting a ``checkpoint_location``, Spark Structured Streaming will store progress
          information and intermediate state, enabling recovery after failures.
          This parameter is only supported when the argument ``df``
          is a streaming :class:`DataFrame <pyspark.sql.DataFrame>`.
       :param trigger: ``trigger`` defines the timing of stream data processing,
          the dictionary will be unpacked and passed to :meth:`DataStreamWriter.trigger <pyspark.sql.streaming.DataStreamWriter.trigger>` as arguments.
          For example, ``trigger={'once': True}`` will result in a call to ``DataStreamWriter.trigger(once=True)``.
       :return: A PySpark :class:`StreamingQuery <pyspark.sql.streaming.StreamingQuery>`.
    """
    fs = FeatureStoreClient()

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    def compute_and_write(
        input: Dict[str, Any], feature_table_name: str, mode: str = MERGE
    ) -> DataFrame:
        """Compute and write to a df."""
        df = func(**input) if isinstance(input, dict) else func(input)
        fs.write_table(
            feature_table_name,
            df,
            mode,
        )
        return df

    def compute_and_write_streaming(
        input: Dict[str, Any],
        feature_table_name: str,
        checkpoint_location: Union[str, None] = None,
        trigger: Dict[str, Any] = FeatureStoreClient._DEFAULT_WRITE_STREAM_TRIGGER,
    ) -> StreamingQuery:
        """Compute and write a streaming df."""
        df = func(**input) if isinstance(input, dict) else func(input)
        if not df.isStreaming:
            raise ValueError(
                "compute_and_write_streaming expects the decorated function to return a streaming DataFrame, "
                "received a non-streaming DataFrame."
            )
        streaming_query = fs.write_table(
            feature_table_name,
            df,
            MERGE,
            checkpoint_location=checkpoint_location,
            trigger=trigger,
        )
        return streaming_query

    wrapper.compute_and_write = compute_and_write
    wrapper.compute_and_write_streaming = compute_and_write_streaming
    return wrapper
