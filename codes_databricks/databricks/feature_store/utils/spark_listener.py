import sys
import uuid
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils import databricks_utils
from py4j.java_gateway import CallbackServerParameters
from pyspark import SparkContext

import logging

_JAVA_PACKAGE = "org.mlflow.spark.autologging"

_logger = logging.getLogger(__name__)


def _get_spark_major_version(sc):
    spark_version_parts = sc.version.split(".")
    spark_major_version = None
    if len(spark_version_parts) > 0:
        spark_major_version = int(spark_version_parts[0])
    return spark_major_version


def _get_jvm_event_publisher():
    """
    Get JVM-side object implementing the following methods:
    - init() for initializing JVM state needed for attaching a SparkListener to watch for datasource
    - register(subscriber) for registering subscribers to receive datasource events
    """
    jvm = SparkContext._gateway.jvm
    qualified_classname = "{}.{}".format(_JAVA_PACKAGE, "MlflowAutologEventPublisher")
    return getattr(jvm, qualified_classname)


def _get_repl_id():
    """
    Get a unique REPL ID for a PythonSubscriber instance. This is used to distinguish between
    REPLs in multitenant, REPL-aware environments where multiple Python processes may share the
    same Spark JVM (e.g. in Databricks). In such environments, we pull the REPL ID from Spark
    local properties, and expect that the PythonSubscriber for the current Python process only
    receives events for datasource reads triggered by the current process.
    """
    repl_id = databricks_utils.get_repl_id()
    if repl_id:
        return repl_id
    main_file = sys.argv[0] if len(sys.argv) > 0 else "<console>"
    return "PythonSubscriber[{filename}][{id}]".format(
        filename=main_file, id=uuid.uuid4().hex
    )


class PythonSubscriber(object):
    """
    Subscriber, intended to be instantiated once per Python process, that subscribes to Spark for
    information from JVM about the spark data sources read during the lifetime of this subscriber.
    This class implements Java interface (org.mlflow.spark.autologging.MlflowAutologEventSubscriber,
    defined in the mlflow-spark package) that's called-into by autologging logic in the JVM in order
    to propagate Spark datasource read events to Python.

    This class leverages the Py4j callback mechanism to receive callbacks from the JVM, see
    https://www.py4j.org/advanced_topics.html#implementing-java-interfaces-from-python-callback for
    more information.
    """

    def __init__(self):
        self._repl_id = _get_repl_id()
        self._data_sources = []
        self._registered = False
        # Internal message for client, if needed. Can be externalized in future.
        self._message = None

    def toString(self):
        # For debugging
        return "PythonSubscriber<replId=%s>" % self.replId()

    def ping(self):
        return None

    def notify(self, path, version, data_format):
        # Record data sources only if this subscriber is registered.
        if not self.is_active():
            _logger.warning(f"Data source listener not active. Cause: ${self._message}")
        else:
            self._data_sources.append((path, version, data_format))

    def register(self):
        event_publisher = _get_jvm_event_publisher()
        event_publisher.register(self)
        self._registered = True

    def replId(self):
        return self._repl_id

    def get_data_sources(self):
        source_type_to_data_source = {}
        for (path, _, fmt) in self._data_sources:
            if fmt not in source_type_to_data_source:
                source_type_to_data_source[fmt] = []
            source_type_to_data_source[fmt].append(path)
        return source_type_to_data_source

    class Java:
        implements = ["{}.MlflowAutologEventSubscriber".format(_JAVA_PACKAGE)]

    def disable(self):
        self._registered = False

    def set_error(self, msg):
        self._message = msg

    def is_active(self):
        return self._registered


def spark_activity_subscriber(spark_context):
    spark_table_info_listener = PythonSubscriber()
    if _get_spark_major_version(spark_context) < 3:
        msg = "Data source listener disabled. Use Spark version >= 3"
        _logger.warning(msg)
        spark_table_info_listener.set_error(msg)
        return spark_table_info_listener
    gw = spark_context._gateway
    params = gw.callback_server_parameters
    callback_server_params = CallbackServerParameters(
        address=params.address,
        port=params.port,
        daemonize=True,
        daemonize_connections=True,
        eager_load=params.eager_load,
        ssl_context=params.ssl_context,
        accept_timeout=params.accept_timeout,
        read_timeout=params.read_timeout,
        auth_token=params.auth_token,
    )
    callback_server_started = gw.start_callback_server(callback_server_params)

    try:
        event_publisher = _get_jvm_event_publisher()
        event_publisher.init(1)
        spark_table_info_listener.register()
    except Exception as e:
        if callback_server_started:
            try:
                gw.shutdown_callback_server()
            except Exception as e:  # pylint: disable=broad-except
                _logger.warning(
                    "Failed to shut down Spark callback server for autologging: %s",
                    str(e),
                )
        spark_table_info_listener.set_error("Failed to start callback server.")
        spark_table_info_listener.disable()
    if not spark_table_info_listener.is_active() or not _get_active_spark_session():
        _logger.warning(
            "Exception while attempting to initialize JVM-side state for logging "
            "Spark datasource. Either use a cluster with ML Runtime or install "
            "the mlflow-spark JAR in your cluster as described in "
            "http://mlflow.org/docs/latest/tracking.html#automatic-logging-from-spark-experimental."
        )
    return spark_table_info_listener


class SparkSourceListener(object):
    def __init__(self):
        pass

    def __enter__(self):
        active_session = _get_active_spark_session()
        if active_session is not None:
            # We know SparkContext exists here already, so get it
            sc = SparkContext.getOrCreate()
            return spark_activity_subscriber(sc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
