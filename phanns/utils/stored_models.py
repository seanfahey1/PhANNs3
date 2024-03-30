from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import shutil
from datetime import datetime
from shutil import rmtree


def get_model_dir(name):
    saved_model_dir = Path(__file__).parent.parent.parent / f"model_files/{name}/"
    saved_model_dir.mkdir(exist_ok=True, parents=True)
    return saved_model_dir


def load_stored_model(name: str):
    saved_model_dir = get_model_dir(name)

    with open(saved_model_dir / "arrays/arr.parquet", 'rb') as file:
        parquet_table = pq.read_table(file)
        mean_arr = np.array(parquet_table['mean'])
        std_arr =np.array(parquet_table['std'])

    models_dir = saved_model_dir / "model_files/"
    model_paths = [str(x.absolute) for x in models_dir.glob("*.keras")]

    return model_paths, mean_arr, std_arr


def store_newly_generated_model(name: str, std_arr: np.array, mean_arr: np.array):
    saved_model_dir = get_model_dir(name)
    array_dir = saved_model_dir / 'arrays'
    model_dir = saved_model_dir / 'model_files'
    array_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    parquet_table = pa.table(
        {
            "std": std_arr,
            "mean": mean_arr
        }
    )

    pq.write_table(parquet_table, array_dir / "arr.parquet")

    pass


def store_model_from_disk(name, model_dir):
    saved_model_dir = Path(__file__).parent / f"model_files/{name}/"
    saved_model_dir.mkdir(exist_ok=True, parents=True)

    new_model_dir = saved_model_dir / f"{name}/"
    shutil.copytree(model_dir, new_model_dir, dirs_exist_ok=True)


def list_models():
    models_dir = Path(__file__).parent.parent.parent / f"model_files/"
    available_models = [x for x in models_dir.glob("*/")]
    for model in available_models:
        timestamp = str(datetime.fromtimestamp(model.stat().st_mtime))
        print(f"model name: {model.stem: <20}last edited: {timestamp}")


def remove_model(name):
    model_dir = str(get_model_dir(name).absolute)
    rmtree(model_dir)
