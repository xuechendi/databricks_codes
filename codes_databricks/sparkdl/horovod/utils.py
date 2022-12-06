# Copyright 2018 Databricks, Inc.
#
# pylint: disable=too-few-public-methods
# pylint: disable=invalid-name

import errno
import io
import logging
import os
import pwd
import shutil
import subprocess
import tempfile
import textwrap
import time

import paramiko

from sparkdl.utils import logwrap, _get_max_num_concurrent_tasks


class IOUtils(object):
    """
    Utility class to manage directories and files.
    """
    @staticmethod
    def makedirs_exist_ok(name):
        """
        Recursively creates directories, and skips if directory already exists.
        This is similar to os.makedirs(name, exist_ok=True) in Python 3.
        :param name: dir name
        """
        try:
            os.makedirs(name)
        except OSError:
            # Tasks on the same worker would attempt to create the same dir.
            # In case of error, we just need to confirm the dir already exists.
            if not os.path.isdir(name):
                raise


class SshSessionManager(object):
    """
    Helper class to manage an SSH session per run.
    It assumes the SSH setup is for the current user.
    """
    def __init__(self, run_dir, _home_dir=None, _configure_ssh=True):
        """
        :param run_dir: working dir for the run, where we write the private key
        :param _home_dir: home directory (for test only)
        :param _configure_ssh: whether to configure SSH (for test only)
        """
        self._run_dir = run_dir
        self._home_dir = _home_dir or SshSessionManager._get_home_dir()
        self._configure_ssh = _configure_ssh
        self._generate_keypair()
        self.private_key_path = os.path.join(self._run_dir, "id_rsa")

    @staticmethod
    def _get_home_dir():
        """
        Returns the home dir of the current user.
        """
        return pwd.getpwuid(os.getuid()).pw_dir

    def _generate_keypair(self):
        """
        Generates and stores an SSH key pair.
        """
        pair = paramiko.RSAKey.generate(2048)
        private_key_out = io.StringIO()
        pair.write_private_key(private_key_out)
        self._private_key = private_key_out.getvalue()
        self._public_key = "ssh-rsa %s" % pair.get_base64()

    def write_private_key(self):
        """
        Writes the private key under the run dir and returns its path.
        This method is called by the driver and the first executor.
        """
        if not self._configure_ssh:
            return None
        assert os.path.isdir(self._run_dir)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(self._private_key)
        shutil.move(f.name, self.private_key_path)
        os.chmod(self.private_key_path, 0o400)
        return self.private_key_path

    def _get_public_key_entry(self):
        """Returns the public key entry to be added to authorized_keys."""
        return textwrap.dedent("""
        # Added by HorovodRunner
        {}
        """.format(self._public_key))

    def authorize_public_key(self):
        """
        Adds the public key to authorized_keys.
        This method is called by all executors.
        """
        if not self._configure_ssh:
            return
        ssh_dir = os.path.join(self._home_dir, ".ssh")
        IOUtils.makedirs_exist_ok(ssh_dir)
        authorized_keys_path = os.path.join(ssh_dir, "authorized_keys")
        entry = self._get_public_key_entry()
        with open(authorized_keys_path, 'a') as f:
            # We assume the underlying file system is POSIX-compliant.
            # The content is less than 512 bytes and hence less than PIPE_BUF.
            # So the append should be atomic without explict locks.
            # See write(2) and pipe(7).
            f.write(entry)

    def mask_authorized_public_key(self):
        """
        Masks the public key previously added to authorized_keys by "#"s of the same length.
        So the offsets of the key entries are not changed.
        This method is called by all executors after each run.
        """
        if not self._configure_ssh:
            return
        authorized_key_path = os.path.join(self._home_dir, ".ssh", "authorized_keys")
        target = self._public_key + "\n"
        mask = "#" * len(self._public_key) + "\n"
        with open(authorized_key_path, "r+") as f:
            # We do not use "for line in f:" because it has a read-ahead buffer.
            # See https://docs.python.org/2/library/stdtypes.html#file.next
            pos = f.tell()
            line = f.readline()
            while line:
                if line == target:
                    f.seek(pos)
                    f.write(mask)
                pos = f.tell()
                line = f.readline()

    def clean_private_key(self):
        """
        Clean up the private key under the run dir.
        This method is called by the driver and the first executor.
        """
        if not self._configure_ssh:
            return
        try:
            os.remove(self.private_key_path)
        except OSError as ex:
            # In CE and local mode, there are two delete attempts,
            # one from driver and one from worker #0.
            if ex.errno != errno.ENOENT:
                raise

    @staticmethod
    def setup_sshd():
        """
        Start the sshd if it is not running
        """
        # "service ssh status" will return 0, if ssh is running.
        if os.system("service ssh status"):
            logging.warning("SSH is not running. Start SSH service.")
            os.system("service ssh start")

def inherit_doc(cls):
    """
    A decorator that makes a class inherit documentation from its parents.
    """
    for name, func in vars(cls).items():
        # only inherit docstring for public functions
        if name != "__init__" and name.startswith("_"):
            continue
        if not func.__doc__:
            for parent in cls.__bases__:
                parent_func = getattr(parent, name, None)
                if parent_func and getattr(parent_func, "__doc__", None):
                    func.__doc__ = parent_func.__doc__
                    break
    cls.__doc__ = cls.__bases__[0].__doc__
    return cls


def check_and_get_num_partitions(sc, timeout_seconds=2 * 60):
    """
    Gets total number of partitions possible concurrently on a given spark context and raises
    an exception if there are no executors in timeout_seconds.
    """
    WARN_INTERVAL = 5
    CHECK_INTERVAL = 0.1
    current_time = time.time()
    start_time = current_time
    warn_time = current_time
    end_time = current_time + timeout_seconds
    while current_time < end_time:
        num_partitions = _get_max_num_concurrent_tasks(sc)
        if num_partitions > 0:
            return num_partitions
        if current_time >= warn_time:
            logging.warning(logwrap.fill("""
                Waiting for executor registration before HorovodRunner could start.
                {} seconds / {} seconds
                """.format(int(current_time-start_time), timeout_seconds)))
            warn_time += WARN_INTERVAL
        time.sleep(CHECK_INTERVAL)
        current_time = time.time()

    raise RuntimeError(logwrap.fill("""
        The cluster has no workers. At least 1 worker is needed to use
        HorovodRunner with np=0. Please increase your cluster size and retry.
        """))


def get_num_gpus():
    """
    Returns the number of gpus on a machine.
    """
    # This is necessary on GCP because nvidia-smi is in /usr/local/nvidia/bin,
    # which is not on the PATH. It has no effect on other clouds.
    set_path = "PATH=/usr/local/nvidia/bin:$PATH"
    result = subprocess.run(["bash", "-c", f"{set_path} nvidia-smi -L | wc -l"],
                            capture_output=True, check=False)
    # On CPU instances, nvidia-smi is unavailable. wc receives empty stdout and returns 0.
    num_gpus = int(result.stdout)
    return num_gpus


def get_gpu_amount_per_task(sc):
    return sc.getConf().get("spark.task.resource.gpu.amount")


def get_slots_per_partition(sc):
    """
    Returns number of slots we can have per partition, i.e.,
    return number of spark config value "spark.task.resource.gpu.amount" on a GPU cluster,
    or return 1 on a CPU cluster.
    If the cluster has no workers, it will raise a runtime error.
    """
    check_and_get_num_partitions(sc)
    gpu_per_task_str = get_gpu_amount_per_task(sc)
    if gpu_per_task_str is None:
        # not on a gpu cluster
        return 1
    else:
        return int(gpu_per_task_str)
