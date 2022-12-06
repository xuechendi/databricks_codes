# Copyright 2018 Databricks, Inc.
#
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

from __future__ import absolute_import, division, print_function

import collections
import ctypes
import inspect
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from collections import defaultdict

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.taskcontext import BarrierTaskContext
from pyspark import cloudpickle
try:
    # for cloudpickle in pyspark 3.1.0
    from pyspark.cloudpickle.cloudpickle import _extract_code_globals as extract_code_globals
except ImportError:
    # for backward compatibility with cloudpickle in pyspark 3.0.1
    from pyspark.cloudpickle import CloudPickler
    extract_code_globals = CloudPickler.extract_code_globals

from sparkdl import utils
from sparkdl.horovod.utils import get_num_gpus
from sparkdl.utils import logwrap
from sparkdl.utils.instrumentation import instrumented
from .runner_base import HorovodRunnerBase
from .log_communication import get_driver_host, LogStreamingClient, LogStreamingServer
from .utils import SshSessionManager, IOUtils, inherit_doc, get_slots_per_partition, \
    check_and_get_num_partitions, get_gpu_amount_per_task

_PICKLED_FUNC_FILENAME = "func.pkl"
_PICKLED_RESULT_FILENAME = "result.pkl"
_PYTHON_PATH_FILENAME = "python_path.txt"
_CUDA_VISIBLE_DEVICES_FILENAME = "cuda_visible_devices.txt"
_LAUNCHER_FILENAME = "launch.sh"
_HOROVOD_TIMELINE = "HOROVOD_TIMELINE"
_PICKED_FUNC_WARN_SIZE = 10 * 1024 * 1024
_TAIL_LINES_TO_KEEP = 100


def _get_random_id():
    """
    Generates a random ID.
    """
    return uuid.uuid4().hex[-12:]


def _is_first_task_on_local_node(task_ip_list, partition_id):
    task_ip = task_ip_list[partition_id]
    return task_ip not in task_ip_list[:partition_id]


def _get_cuda_visible_devices_env():
    return os.getenv('CUDA_VISIBLE_DEVICES')


def _gen_node_ip_to_gpu_set_map(cuda_visible_devices_str_list, task_ip_list):
    node_ip_to_gpu_set_map = defaultdict(set)
    for partition_id, gpu_list_str in enumerate(cuda_visible_devices_str_list):
        node_ip = task_ip_list[partition_id]
        gpu_set = {int(gpu_addr) for gpu_addr in gpu_list_str.split(',')}
        node_gpu_set = node_ip_to_gpu_set_map[node_ip]
        assert len(node_gpu_set & gpu_set) == 0, 'Different tasks should not share gpu.'
        node_gpu_set.update(gpu_set)
    return node_ip_to_gpu_set_map


def _get_hvd_worker_cuda_visible_devices_str(
        use_gpu,
        partition_id,
        node_ip_to_gpu_set_map,
        task_ip_list):
    if use_gpu:
        task_ip = task_ip_list[partition_id]
        gpu_set = node_ip_to_gpu_set_map[task_ip]
        return ','.join(map(str, sorted(list(gpu_set))))
    else:
        return ''


class MpiProcessMapper():
    """
    Helper class that maps MPI processes to Spark task slots. It manages number of slots per
    partition, processes and mapping between those. This class gets instantiated on the driver and
    is sent to all worker nodes. Only the rank 0 process uses it to understand how to distribute
    processes on hosts. Say we have
        * np = 10,
        * slots_per_partition = 4 (gpus),
    then we will have
        * ranks_for_partition @(0) -> [0, 3, 6, 9], @(1) -> [1, 4, 7], @(2) -> [2, 5, 8]
        * hosts_list_for_mpi = ["h0", "h1", "h2", "h0", "h1", "h2", "h0", "h1", "h2", "h0"]
        where "h_i" are hostnames for the i-th Spark partition
    """

    def __init__(self, np, slots_per_partition):
        """
        :param np: Number of MPI processes we want to launch
        :param slots_per_partition: Number of MPI processes we can launch per Spark partition
        (task slot). On Databricks, this should be 1 for cpu clusters and number of GPUs for
        gpu-enabled clusters
        """
        self.np = np
        self.slots_per_partition = slots_per_partition

    @property
    def num_partitions(self):
        """Gets minimal number of Spark partitions required to launch `np` MPI processes"""
        return int(math.ceil(self.np / self.slots_per_partition))

    def ranks_for_partition(self, partition_id):
        """
        Gets MPI ranks allocated to a specific Spark partition. This function is called while
        preparing working directories on a partition to understand which mpi ranks will be run on
        a given host.
        """
        assert partition_id < self.num_partitions
        return list(range(partition_id, self.np, self.num_partitions))

    def hosts_list_for_mpi(self, hosts):
        """
        Gets the hosts list for MPI command, ordered by the ranks of MPI processes that shall run
        on them. This function is called by the partition_id==0 process to spawn mpi processes
        """
        assert len(hosts) == self.num_partitions
        return (hosts * self.slots_per_partition)[:self.np]

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, str(self.__dict__))


@inherit_doc
class HorovodRunner(HorovodRunnerBase):

    def __init__(  # pylint: disable=super-init-not-called
            self,
            *,
            np,
            driver_log_verbosity="log_callback_only"):
        self.num_processor = np
        self.python_executable = sys.executable
        self.logger = utils.get_logger(self.__class__.__name__)
        self._last_root_working_dir = None  # Stores the working directory for debugging
        # Parameters exposed for testing
        self._configure_ssh = True  # whether to configure SSH per run
        self._debug = False  # whether to leave SSH keys and working directory after run

        if driver_log_verbosity not in ["all", "log_callback_only"]:
            raise ValueError(
                "driver_log_verbosity must be 'log_callback_only' or 'all' "
                f"but got {driver_log_verbosity}.")

        self._stream_all_logs = driver_log_verbosity == "all"
        if not self._stream_all_logs:
            self.logger.warning(logwrap.fill("""
            HorovodRunner will only stream logs generated by :func:`sparkdl.horovod.log_to_driver`
            or :class:`sparkdl.horovod.tensorflow.keras.LogCallback` to notebook cell output.
            If want to stream all logs to driver for debugging, you can set driver_log_verbosity
            to 'all', like `HorovodRunner(np=2, driver_log_verbosity='all')`.
            """))

        if np == 0:
            self.logger.warning(logwrap.fill("""
                Setting np=0 is deprecated and it will be removed in the next major Databricks Runtime release.
                Choosing np based on the total task slots at runtime is unreliable due to dynamic executor registration.
                Please set the number of parallel processes you need explicitly."""))

        if self.num_processor < 0:
            self.logger.warning(logwrap.fill(
                "HorovodRunner will launch Horovod jobs on the driver node. "
                "There would be resource contention if you share the cluster with others."))
            self.sc = None
        else:
            spark = SparkSession.builder.getOrCreate()
            self.sc = spark.sparkContext

    @instrumented
    def run(self, main, **kwargs):
        run_id = self.__class__.__name__ + "_" + _get_random_id()

        self._check_method_args(main, kwargs)

        log_streaming_server = LogStreamingServer()
        if self.num_processor < 0:
            # local mode, avoid to create spark context. Use local IP.
            driver_address = '127.0.0.1'
        else:
            driver_address = get_driver_host(self.sc)

        log_streaming_server.start()
        time.sleep(1)  # wait server starting
        log_streaming_server_port = log_streaming_server.port

        # Invokes main with kwargs. So we don't need to pickle them separately.
        def wrapped_main(rank=0):
            # This is only necessary when running on GPU nodes in GCP
            # It is a no-op in other environments
            # pylint: disable=subprocess-run-check
            subprocess.run(["ldconfig", "/usr/local/nvidia/lib64"])

            # Configure the root logger before user code. So user code can modify it.
            utils.get_logger(name=None)

            # Initialize the log client and create the instance.
            # So HorovodRunner log callbacks can use the client.
            LogStreamingClient._init(driver_address, log_streaming_server_port)
            try:
                return_value = main(**kwargs)
            finally:
                try:
                    LogStreamingClient._destroy()
                except BaseException:
                    pass

            if rank == 0:
                with open(_PICKLED_RESULT_FILENAME, 'wb') as f:
                    try:
                        cloudpickle.dump(return_value, f)
                    except Exception as e:
                        raise RuntimeError("Caught an exception while pickling "
                                           "return value: {}".format(repr(e)))

        self._log_global_vars(main)
        pickled_func_str = cloudpickle.dumps(wrapped_main)
        self._check_pickled_function(pickled_func_str)

        root_working_dir = os.path.join(tempfile.gettempdir(), run_id)
        self._last_root_working_dir = root_working_dir

        try:
            return self._run_program(root_working_dir, pickled_func_str, log_streaming_server_port)
        finally:
            log_streaming_server.shutdown()

    @staticmethod
    def _get_main_result_bytes(root_working_dir):
        """
        Call on the same partition/process that main function is evaluated at.
        This assumes that `_run_command` has been called before and `_PICKLED_RESULT_FILENAME`
        has been written.
        """
        with open(os.path.join(root_working_dir, _PICKLED_RESULT_FILENAME), 'rb') as f:
            return f.read()

    @staticmethod
    def _parse_result_bytes(result_bytes):
        try:
            return cloudpickle.loads(result_bytes)
        except Exception as e:
            raise RuntimeError("Caught an excpetion while unpickling "
                               "return value: {}".format(repr(e)))

    @staticmethod
    def _run_command(args, _prctl=True, redirect_to_stdout=True, log_streaming_client=None):
        """
        Runs a command in a new process, redirects stdout/stderr to stdout, and handles termination
        of the command process.

        :param args: a list of command arguments
        :param _prctl: (test only) use prctl to signal the command process upon parent death
        :param redirect_to_stdout: whether to redirect all stdout/stderr logs to current stdout
        :param log_streaming_client: if not None, also stream logs via the client
        """
        def sigterm_on_parent_death():
            """
            Uses prctl to automatically send SIGTERM to the command process when its parent is dead.

            This handles the case when the parent is a PySpark worker process.
            If a user cancels the PySpark job, the worker process gets killed, regardless of
            PySpark daemon and worker reuse settings.
            We use prctl to ensure the command process receives SIGTERM after job cancellation.
            The command process itself should handle SIGTERM properly, which is true for `mpirun`.
            This is a no-op on macOS because prctl is not supported.

            """
            if _prctl:
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    # Set the parent process death signal of the command process to SIGTERM.
                    libc.prctl(1,  # PR_SET_PDEATHSIG, see prctl.h
                               signal.SIGTERM)
                except OSError:
                    pass
        logger = utils.get_logger(HorovodRunner.__name__)
        logger.info("Executing command: {args}.\n".format(args=args))
        task = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE,
                                env=os.environ,  # preserve the environ to support nested MPI jobs
                                preexec_fn=sigterm_on_parent_death)
        task.stdin.close()
        tail = collections.deque(maxlen=_TAIL_LINES_TO_KEEP)
        try:
            for line in task.stdout:
                decoded = line.decode()
                tail.append(decoded)
                if redirect_to_stdout:
                    sys.stdout.write(decoded)
                if log_streaming_client:
                    log_streaming_client.send(decoded.rstrip())
            task.wait()
        finally:
            if task.poll() is None:
                try:
                    task.terminate()  # SIGTERM
                    time.sleep(0.5)
                    if task.poll() is None:
                        task.kill()  # SIGKILL
                except OSError:
                    pass
        if task.returncode != os.EX_OK:
            if len(tail) == _TAIL_LINES_TO_KEEP:
                last_n_msg = "last %d lines of the task output are" % _TAIL_LINES_TO_KEEP
            else:
                last_n_msg = "task output is"
            raise RuntimeError(
                "Command %s failed with return code %d.\n" % (args, task.returncode) +
                logwrap.fill("""
                    The %s included below. If you're having trouble accessing the full output,
                    check the executor logs for Task 0 in the Spark UI.
                    """ % last_n_msg) +
                "\n%s\n" % "".join(tail))

    @staticmethod
    def _get_mpi_command(hosts, args, extra_mpi_args, private_key_path):
        """
        Builds the MPI command, called by the first executor.
        :param hosts: a list of worker IPs
        :param args: command for the MPI worker process
        :param extra_mpi_args: extra MPI args
        :param private_key_path: path to the private key for SSH
        :return: a full list of args to launch the MPI job
        """
        cmd = ["mpirun", "--allow-run-as-root"]
        np = len(hosts)
        cmd += ["-np", np]
        hosts_str = ','.join(hosts)
        cmd += ["-H", hosts_str]
        # [ES-8851] Do not read from stdin because it might inherit PySpark worker daemon's stdin,
        # which is used to receive the PID of PySpark worker process to terminate.
        cmd += ["--stdin", "none"]
        # Prepend each line of output with [jobid,rank]<stdxxx>.
        cmd += ["--tag-output"]
        # We use sequential mapper so MPI process i always runs on hosts[i].
        # See https://www.open-mpi.org/doc/v3.0/man1/mpirun.1.php#sect9
        # The slot counts don't matter. So we do not need "--map-by slot".
        cmd += ["-mca", "rmaps", "seq"]
        # Do not bind a training process to a single CPU core.
        cmd += ["--bind-to", "none"]
        # Have NCCL print debugging information.
        cmd += ["-x", "NCCL_DEBUG=INFO"]
        # The following args are recommended in the official Horovod doc.
        # See https://github.com/uber/horovod/blob/master/docs/running.md.
        # It forces the use of TCP for MPI communication, instead of RDMA.
        # This is to avoid multiprocessing issues with RDMA that result segfault.
        # However, at least for now, it should be a no-op for us.
        # Only Azure provides RDMA support on limited VM types:
        # https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes-hpc#rdma-capable-instances
        # And we need to configure LXC container to support RDMA.
        # See https://community.mellanox.com/docs/DOC-2965.
        # We still use those args for future-proof.
        cmd += ["-mca", "pml", "ob1", "-mca", "btl", "^openib"]
        # Disable strict host key checking for SSH.
        ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if private_key_path is not None:
            ssh_args += ["-i", private_key_path]
        cmd += ["-mca", "plm_rsh_agent", " ".join(ssh_args)]
        cmd += extra_mpi_args
        cmd += args
        cmd = list(map(str, cmd))
        return cmd

    def _check_encryption(self):

        if self.num_processor >= 0:
            sql_context = SQLContext(self.sc)
            isSslEnabled = utils._getConfBoolean(sql_context, "spark.ssl.enabled", "false")
            ignoreSsl = utils._getConfBoolean(sql_context, "databricks.horovod.ignoreSsl", "false")

            if isSslEnabled and ignoreSsl:
                self.logger.warning(logwrap.fill("""
                    This cluster has TLS encryption enabled; however, {name} does not
                    support data encryption in transit. The Spark configuration 
                    'databricks.hovorod.ignoreSsl' has been set to true to override this 
                    configuration and use {name} anyway. Please note this will cause model 
                    parameters and possibly training data to be sent between nodes unencrypted.
                    """.format(name=self.__class__.__name__)))
            elif isSslEnabled:
                raise Exception(logwrap.fill("""
                This cluster has TLS encryption enabled; however, {name} does not support 
                data encryption in transit. To override this configuration and use {name} 
                anyway, you may set 'databricks.hovorod.ignoreSsl' to true in the Spark 
                configuration. Please note this will cause model parameters and possibly training 
                data to be sent between nodes unencrypted.""".format(
                    name=self.__class__.__name__)))

        return

    def _check_and_get_num_procs(self, slots_per_partition):
        """
        Check the user-set value of num_processor vs the number of available executors.
        This should only be used when num_processor is set to >= 0 (distributed mode).
        :param slots_per_partition: Number of MPI processes we can launch per Spark partition
        (task slot). On Databricks, this should be 1 for cpu clusters and number of GPUs for
        gpu-enabled clusters
        :return: number of tasks to use in the barrier job.  This differs from the user-set value
         if the user sets 0 (all).
        """
        num_procs = self.num_processor

        num_partitions = check_and_get_num_partitions(self.sc)  # also asserts num_partitions > 1
        max_procs = slots_per_partition * num_partitions

        # calculate the results
        if num_procs == 0:
            num_procs = max_procs
        elif num_procs % slots_per_partition != 0:
            # this won't happen for CPU nodes because slots_per_partition is 1
            self.logger.warning(logwrap.fill("""
            HorovodRunner was called with np={} which is not a multiple of the number of GPUs 
            available on each worker, {}. To fully utilize workers set np to be a multiple of {} or
            set np=0 to fully utilize all workers. 
            """.format(num_procs, slots_per_partition, slots_per_partition)))

        if num_procs > max_procs:
            self.logger.warning(logwrap.fill("""
            HorovodRunner was called with np={}, which is greater than the maximum processes that
            can be placed on this cluster. This cluster can place at most {} processes on {}
            executors. Training won't start until there are enough workers on this cluster. You 
            can increase the cluster size or cancel the current run and retry with a smaller np.
            """.format(num_procs, max_procs, num_partitions)))

        return num_procs

    def _run_program(self, root_working_dir, pickled_func_str, log_streaming_server_port):
        _debug = self._debug
        timeline_path = os.getenv(_HOROVOD_TIMELINE, None)
        if timeline_path is None:
            self.logger.info(logwrap.fill(title="How to enable Horovod Timeline?",
                                          text="""
            HorovodRunner has the ability to record the timeline of its activity with Horovod 
            Timeline. To record a Horovod Timeline, set the `HOROVOD_TIMELINE` environment variable 
            to the location of the timeline file to be created. You can then open the timeline file 
            using the chrome://tracing facility of the Chrome browser.
            """))

        args = ["bash", os.path.join(root_working_dir, _LAUNCHER_FILENAME)]
        if self.num_processor < 0:
            local_np = -self.num_processor
            num_gpus = get_num_gpus()
            # if num_gpus == 0, cuda_visible_devices_str will be empty.
            if num_gpus == 0:
                cuda_visible_devices_str = ''
            else:
                cuda_visible_devices_str = ','.join(list(map(str, range(num_gpus))))
            for i in range(local_np):
                HorovodRunner._prepare_working_dir(root_working_dir, i, pickled_func_str,
                                                   cuda_visible_devices_str)
            hosts = ["localhost"] * local_np
            ssh_session = SshSessionManager(root_working_dir, _configure_ssh=self._configure_ssh)
            ssh_session.write_private_key()
            ssh_session.authorize_public_key()
            try:
                # We don't need set HOROVOD_TIMELINE in extra_mpi_args
                # because the MPI process will inherit env vars from parent.
                cmd = HorovodRunner._get_mpi_command(
                    hosts, args, extra_mpi_args=[], private_key_path=ssh_session.private_key_path)
                HorovodRunner._run_command(
                    cmd,
                    redirect_to_stdout=self._stream_all_logs)
                local_bytes = HorovodRunner._get_main_result_bytes(root_working_dir)
                result = HorovodRunner._parse_result_bytes(local_bytes)
            finally:
                if not _debug:
                    ssh_session.clean_private_key()
                    ssh_session.mask_authorized_public_key()
                    shutil.rmtree(root_working_dir, ignore_errors=True)
            return result

        # else num_procs >= 0
        use_gpu = get_gpu_amount_per_task(self.sc) is not None
        slots_per_partition = get_slots_per_partition(self.sc)
        num_procs = self._check_and_get_num_procs(slots_per_partition)

        mapper = MpiProcessMapper(
            np=num_procs,
            slots_per_partition=slots_per_partition)

        self.logger.debug('mapping using: {}'.format(mapper))

        bc_pickled_func_str = self.sc.broadcast(pickled_func_str)
        ssh_session = SshSessionManager(root_working_dir, _configure_ssh=self._configure_ssh)

        IOUtils.makedirs_exist_ok(root_working_dir)

        # Write private key on the driver node.
        # So we can re-run the MPI command on the driver for debugging.
        ssh_session.write_private_key()

        driver_address = get_driver_host(self.sc)

        stream_all_logs = self._stream_all_logs

        def run_horovod_task(_):
            # pylint: disable=too-many-branches
            context = BarrierTaskContext.get()
            partition_id = context.partitionId()
            task_ip_list = [info.address.split(":")[0] for info in context.getTaskInfos()]

            ssh_session.setup_sshd()
            if _is_first_task_on_local_node(task_ip_list, partition_id):
                ssh_session.authorize_public_key()

            node_ip_to_gpu_set_map = None
            if use_gpu:
                task_cuda_visible_devices_str = _get_cuda_visible_devices_env()
                cuda_visible_devices_str_list = context.allGather(task_cuda_visible_devices_str)
                node_ip_to_gpu_set_map = _gen_node_ip_to_gpu_set_map(
                    cuda_visible_devices_str_list,
                    task_ip_list)
            for rank in mapper.ranks_for_partition(partition_id):
                cuda_visible_devices_str = _get_hvd_worker_cuda_visible_devices_str(
                    use_gpu=use_gpu,
                    partition_id=partition_id,
                    node_ip_to_gpu_set_map=node_ip_to_gpu_set_map,
                    task_ip_list=task_ip_list)

                HorovodRunner._prepare_working_dir(
                    root_working_dir,
                    rank,
                    bc_pickled_func_str.value,
                    cuda_visible_devices_str)
            try:
                context.barrier()
                if partition_id == 0:
                    private_key_path = ssh_session.write_private_key()
                    extra_mpi_args = []
                    if timeline_path is not None:
                        os.environ[_HOROVOD_TIMELINE] = timeline_path
                        extra_mpi_args += ['-x', _HOROVOD_TIMELINE]
                    addrs = [info.address.split(":")[0] for info in context.getTaskInfos()]
                    hosts = mapper.hosts_list_for_mpi(addrs)
                    cmd = HorovodRunner._get_mpi_command(
                        hosts=hosts, args=args, extra_mpi_args=extra_mpi_args,
                        private_key_path=private_key_path)
                    log_streaming_client = LogStreamingClient(
                        driver_address, log_streaming_server_port)
                    try:
                        HorovodRunner._run_command(
                            cmd,
                            redirect_to_stdout=True,  # always merge into Spark executor logs
                            log_streaming_client=log_streaming_client if stream_all_logs else None)
                    finally:
                        if not _debug:
                            ssh_session.clean_private_key()
                    partition_bytes = [HorovodRunner._get_main_result_bytes(root_working_dir)]
                else:   # partition != 0
                    partition_bytes = []
                context.barrier()
            finally:
                if not _debug:
                    ssh_session.mask_authorized_public_key()
                    shutil.rmtree(root_working_dir, ignore_errors=True)
            return partition_bytes

        self._check_encryption()

        self.logger.info("Start training.")

        try:
            num_partitions = mapper.num_partitions
            partition_bytes = self.sc.parallelize(range(num_partitions), num_partitions) \
                .barrier() \
                .mapPartitions(run_horovod_task) \
                .collect()
            result = HorovodRunner._parse_result_bytes(partition_bytes[0])
        finally:
            # log_streaming_service.join()
            if not _debug:
                # delete the private key on the driver node.
                ssh_session.clean_private_key()
                # remove the work directory
                shutil.rmtree(root_working_dir, ignore_errors=True)
        return result

    @staticmethod
    def _write_launch_script(root_working_dir):
        """
        Writes a launch script to the root working dir.
        The launch script gets MPI rank from env, loads PYTHONPATH dumped by the PySpark worker,
        and starts a python process to unpickle and invoke the user function.
        :param root_working_dir: the root working dir for the run
        """
        launcher_path = os.path.join(root_working_dir, _LAUNCHER_FILENAME)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(textwrap.dedent(f"""
            set -e
            cd {root_working_dir}
            rank=${{OMPI_COMM_WORLD_RANK:-0}}
            path=$(cat $rank/{_PYTHON_PATH_FILENAME})
            gpu_list=$(cat $rank/{_CUDA_VISIBLE_DEVICES_FILENAME})
            if [[ -n "$gpu_list" ]]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=$gpu_list; fi
            PYTHONPATH=$path {sys.executable} -c "from pyspark import cloudpickle; cloudpickle.load(open('{_PICKLED_FUNC_FILENAME}', 'rb'))(rank=$rank)"
            """))
        shutil.move(f.name, launcher_path)

    @staticmethod
    def _prepare_working_dir(root_working_dir, rank, pickled_func_str, cuda_visible_devices_str):
        """
        Prepares the working directory for a task.
        It saves the pickled function to "func.pkl" under the root directory.
        It saves current PYTHONPATH to root_working_dir/rank/python_path.txt.
        Working directories are not shared among tasks.
        """
        IOUtils.makedirs_exist_ok(root_working_dir)
        HorovodRunner._write_launch_script(root_working_dir)
        working_dir = os.path.join(root_working_dir, str(rank))
        shutil.rmtree(working_dir, ignore_errors=True)
        os.mkdir(working_dir)
        pickled_func_path = os.path.join(root_working_dir, _PICKLED_FUNC_FILENAME)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(pickled_func_str)
        shutil.move(f.name, pickled_func_path)
        cwd = os.getcwd()
        python_path = ':'.join([os.path.join(cwd, i) for i in sys.path])
        python_path_filename = os.path.join(working_dir, _PYTHON_PATH_FILENAME)
        with open(python_path_filename, 'w') as f:
            f.write(python_path)
        cuda_visible_devices_filename = os.path.join(working_dir, _CUDA_VISIBLE_DEVICES_FILENAME)
        with open(cuda_visible_devices_filename, 'w') as f:
            f.write(cuda_visible_devices_str)

    def _log_global_vars(self, main):
        """
        Logs all global names read or written to by the main function to help debugging.
        :param main: the main function provided by the user
        """
        # NOTE: extract_code_globals cannot get the global vars recursively.
        global_variables = extract_code_globals(main.__code__)
        self.logger.info("The global names read or written to by the pickled function are"
                         " %s.", str(global_variables))

    def _check_pickled_function(self, pickled_func_str):
        """
        Checks the size and warns users if it is large.
        :param pickled_func_str: Serialized function string
        """
        pickled_func_size = len(pickled_func_str)
        self.logger.info("The pickled object size is %s bytes.", str(pickled_func_size))
        if pickled_func_size > _PICKED_FUNC_WARN_SIZE:
            self.logger.warning(logwrap.fill("""
            The pickled object size is greater than 10MB. It might cause training slow to start.
            You might consider:
            * Loading large datasets inside the main function instead of materializing
              them on the driver. 
            * Avoid pulling unnecessary variables from the notebook context.
            """))

    def _check_method_args(self, main, kwargs):
        """
        Checks whether the "main" method and the keyword arguments are compatible.
        If not, it logs an ERROR message and raises an exception.
        """
        try:
            inspect.getcallargs(main, **kwargs)     # pylint: disable=deprecated-method
        except TypeError as e:
            self.logger.error(logwrap.fill("""
                The arguments are incompatible with the main function: {}.
                
                The expected signature of the main function is main(**kwargs).
                For example, you may have "def main()" and call "run(main)" without extra arguments,
                or have "def main(steps, rate=0.1)" and then call "run(main, steps=10)",
                or have "def main(steps, rate=0.1, **kwargs)" and then call
                "run(main, steps=10, rate=0.2, debug=True)". 
                """.format(str(e))))
            raise e
