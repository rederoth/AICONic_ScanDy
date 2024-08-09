import pickle
import subprocess
# import rospkg
import tyro
import os
from abc import ABC, abstractmethod


class BaseExperiment(ABC):

    def __init__(self):
        self.is_executed = False
        self.is_loaded = False
        self.directory = None
        self.pre_results = None
        self.results = None
        self.io_threads = []

    @abstractmethod
    def extract_directory(self, config):
        raise NotImplementedError()

    def run(self, config):
        directory = self.extract_directory(config)
        dump_config(config, os.path.join(directory, "config"))
        self.pre_results = self.preaction(config)
        dump_python_object(self.pre_results, os.path.join(directory, "pre_exp_data"))
        self.results = self.experiment_main(config, self.pre_results)
        dump_python_object(self.results, os.path.join(directory, "raw_results"))
        self.postaction(config, self.results, self.pre_results)

    @abstractmethod
    def preaction(self, config):
        raise NotImplementedError()

    @abstractmethod
    def experiment_main(self, config, pre_results):
        raise NotImplementedError()

    def postaction(self, config, results, pre_results):
        for t in self.io_threads:
            t.join()

    def load_exp_data(self, directory):
        self.is_loaded = True
        self.pre_results = load_python_object(os.path.join(directory, "pre_exp_data"))
        self.results = load_python_object(os.path.join(directory, "raw_results"))
        self.directory = directory

    def save_exp_data(self, directory=None):
        if directory is None:
            directory = self.directory
        else:
            self.directory = directory
        dump_python_object(self.pre_results, os.path.join(directory, "pre_exp_data"))
        dump_python_object(self.results, os.path.join(directory, "raw_results"))


def get_git_revision_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


def get_git_revision_short_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


# def get_short_and_long_git_hashes_for_ros_package(package_name):
#     rospack = rospkg.RosPack()
#     package_path = rospack.get_path(package_name)
#     hash = get_git_revision_short_hash(package_path)
#     short_hash = get_git_revision_hash(package_path)
#     return short_hash, hash


def dump_config(config, file_path):
    yaml_str = tyro.to_yaml(config)
    path = file_path+".yaml"
    with open(path, "w") as f:
        f.write(yaml_str)
    return path

def dump_python_object(o, file_path, protocol=4):
    with open(file_path + ".pickle"+str(protocol), "wb") as f:
        pickle.dump(o, f, protocol=protocol)


def load_python_object(file_path, protocol=4):
    with open(file_path + ".pickle"+str(protocol), "rb") as f:
        o = pickle.load(f)
    return o


def get_conda_environment_string(env_name):
    return subprocess.check_output("conda env export --name "+env_name+" | grep -v \"^prefix: \"", shell=True)