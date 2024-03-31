import os
import tarfile

from ...utils.stored_models import store_model


def write_model_files(name, tar_gz_path):
    store_model(name, std_arr, mean_arr, models)


def array_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".parquet":
            yield tarinfo


def model_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".h5":
            yield tarinfo


array_files = []
model_files = []
with tarfile.open("sample.tar.gz") as tar:
    array_files.append(tar.extractall(members=array_files(tar)))
    model_files.append(tar.extractall(members=model_files(tar)))
