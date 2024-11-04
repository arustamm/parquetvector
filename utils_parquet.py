import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib
import os
import dask
import dask.bag as db

@dask.delayed
def read_trace(f, i:int):
    return pa.array(f.trace[i])

def segy_to_parquet(file:pathlib.PosixPath, path_to_save:str, headers: dict, metadata=None):
    import segyio 

    data = []
    fields = []
    with segyio.open(file, "r", ignore_geometry=True) as f:
        for i in range(f.trace.size):
            data.append(read_trace(f, i))
        
        traces = pa.array(list(segyio.tools.collect(f.trace[:])))
        data.append(traces)

        fields.append(pa.field("data", type=traces.type))
        for k, v in headers.items():
            h_arr = f.attributes(v)[:]
            fields.append(pa.field(k, type=pa.from_numpy_dtype(h_arr.dtype)))
            data.append(pa.array(h_arr))

    schema = pa.schema(fields, metadata=metadata)
    table = pa.Table.from_arrays(data, schema=schema)

    base = pathlib.Path(path_to_save)
    base.mkdir(exist_ok=True)
    name, _ = os.path.splitext(file.name)
    path = base / (name + ".parquet")
    with path.open("wb") as sink:
        with pq.ParquetWriter(sink, schema) as writer:
            writer.write_table(table)

def segy_to_parquet_dask(files: list, path_to_save: str, headers: dict, metadata: dict=None, client=None):
    if client is not None:
        dask_client = client
    bag = db.from_sequence(files)
    bag.map(segy_to_parquet, path_to_save, headers, metadata).compute()

def get_files_with_extension(directory: str, extension: str) -> list:
  path = pathlib.Path(directory)
  return list(path.glob(f"*.{extension}"))

