import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from shutil import rmtree

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def get_model_dir(name, create=True):
    saved_model_dir = Path(__file__).parent.parent.parent / f"model_files/{name}/"
    if create:
        saved_model_dir.mkdir(exist_ok=True, parents=True)

    return saved_model_dir


def validate_model(name):
    model_dir = get_model_dir(name, create=False)
    try:
        dirs = [x.stem for x in model_dir.glob("*/")]
        assert "model_files" in dirs
        assert "arrays" in dirs

        model_files = [x.name for x in model_dir.glob("model_files/*.keras")]
        for name in [f"{i:02d}.keras" for i in range(1, 11)]:
            assert name in model_files

        data_files = [x.name for x in model_dir.glob("arrays/*.parquet")]
        assert "arr.parquet" in data_files
        assert "class_names_arr.parquet" in data_files

    except AssertionError as e:
        print(e)
        return False

    return True


def load_stored_model(name: str):
    saved_model_dir = get_model_dir(name)

    with open(saved_model_dir / "arrays/arr.parquet", "rb") as file:
        parquet_table = pq.read_table(file)
        mean_arr = np.array(parquet_table["mean"])
        std_arr = np.array(parquet_table["std"])

    with open(saved_model_dir / "arrays/class_names_arr.parquet", "rb") as file:
        parquet_table = pq.read_table(file)
        sorted_group_names = list(np.array(parquet_table["sorted_group_names"]))

    models_dir = saved_model_dir / "model_files/"
    model_paths = [str(x.absolute) for x in models_dir.glob("*.keras")]

    return model_paths, mean_arr, std_arr, sorted_group_names


def store_newly_generated_model(
    name: str, std_arr: np.array, mean_arr: np.array, sorted_group_names: list
):
    saved_model_dir = get_model_dir(name)
    array_dir = saved_model_dir / "arrays"
    model_dir = saved_model_dir / "model_files"
    array_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    parquet_table = pa.table(
        {
            "std": std_arr,
            "mean": mean_arr,
        }
    )
    group_names_table = pa.table({"sorted_group_names": np.array(sorted_group_names)})

    pq.write_table(parquet_table, array_dir / "arr.parquet")
    pq.write_table(group_names_table, array_dir / "class_names_arr.parquet")

    pass


def store_model_from_disk(name, model_dir):
    saved_model_dir = Path(__file__).parent / f"model_files/{name}/"
    saved_model_dir.mkdir(exist_ok=True, parents=True)

    new_model_dir = saved_model_dir / f"{name}/"
    shutil.copytree(model_dir, new_model_dir, dirs_exist_ok=True)


def list_models():
    models_dir = Path(__file__).parent.parent.parent / f"model_files/"
    available_models = [
        x
        for x in models_dir.glob("*/")
        if not str(x).endswith("<Sequential name=sequential, built=True>")
    ]
    if len(available_models) == 0:
        print("No models are currently saved.")
    else:
        print(f"{'models:': <40}{'time last edited:': <30}{'state'}")
        for model in available_models:
            valid = validate_model(model.name)
            timestamp = str(datetime.fromtimestamp(model.stat().st_mtime))
            if valid:
                print(f"{model.stem: <40}{timestamp: <30}ready")
            else:
                print(f"{model.stem: <40}{timestamp: <30}CORRUPTED!")


def remove_model(name):
    model_dir = str(get_model_dir(name).absolute())
    rmtree(model_dir)


def move_model(old_name, new_name):
    model_dir = get_model_dir(old_name)
    new_model_dir = model_dir.parent / new_name

    new_model_dir.mkdir(parents=True, exist_ok=True)
    model_dir.rename(new_model_dir)


def export_model(name, output_path):
    saved_model_dir = get_model_dir(name)
    model_files = list(
        (saved_model_dir / "model_files").glob("0*")
    )  # TODO: test this change!!
    array_files = list((saved_model_dir / "arrays").glob("*"))

    with tarfile.open(Path(output_path) / f"{name}.tar.gz", "w:gz") as tar:
        print("Writing model files. This step may take some time.")
        for file in tqdm(model_files):
            tar.add(file, arcname=f"model_files/{file.name}")
        print("Writing data files.")
        for file in tqdm(array_files):
            tar.add(file, arcname=f"arrays/{file.name}")


def load_model(name, input_file):
    new_model_dir = get_model_dir(name)

    array_files_dir = new_model_dir / "arrays"
    model_files_dir = new_model_dir / "model_files"

    array_files_dir.mkdir(exist_ok=True, parents=True)
    model_files_dir.mkdir(exist_ok=True, parents=True)

    with tarfile.open(input_file) as tar:
        files = tar.getmembers()
        for file in files:
            file_name = file.get_info()["name"]

            if re.match(r"arrays/.*\.parquet", file_name):
                contents = tar.extractfile(file)
                with open(array_files_dir / file, "wb") as writer:
                    writer.write(contents)

            elif re.match(r"model_files/.*\.parquet", file_name):
                contents = tar.extractfile(file)
                with open(model_files_dir / file, "wb") as writer:
                    writer.write(contents)
