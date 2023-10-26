import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import structlog

logger = structlog.getLogger(__name__)


def _check_memory_usage(path: Path, chunksize: int = 10000) -> None:
    """
    This function provides information regarding how much memory usage
    during reading datset so we can choose the appropriate chunksize
    for reading in batches.
    param path: Path to the location of the dataset
    param chunksize: size of each chunk to read dataset in batches
    return: None
    """
    chunks = pd.read_csv(path, chunksize=chunksize)
    # List to keep track of memory usages of each chunk
    memory_usages = []
    for index, chunk in enumerate(chunks):
        # Get memory usage of the chunk (in bytes)
        mem_usage = chunk.memory_usage(deep=True).sum()
        # Convert memory usage to megabytes for easier reading
        mem_usage_mb = mem_usage / (1024 ** 2)  # Convert bytes to megabytes
        memory_usages.append(mem_usage_mb)
        logger.info(f"Chunk {index + 1} Memory Usage: {mem_usage_mb:.2f} MB")

    # Get the average memory usage across all chunks:
    avg_mem_usage = sum(memory_usages) / len(memory_usages)
    logger.info(f"\nAverage Memory Usage per Chunk: {avg_mem_usage:.2f} MB")


def read_data(path: Path, chunksize: int = None, logging: bool = False) -> pd.DataFrame:
    """
    This function use parameters in the configuration file (config.yml in root folder)
    to read selected dataset and convert them to a Pandas dataframe.
    param path: path to the location of the dataset
    param chunksize: size of each chunk to read dataset in batches
    param logging: whether to see the memory usage for each chunk as well as average memory usage
    return: a Pandas dataframe
    """
    if path.split('.')[-1] == "h5":
        data = {}
        with h5py.File(path, 'r') as file:
            keys = list(file.keys())
            for key in keys:
                data[key] = file[key][:]
        logger.info(f"Finished reading {path}")
        return pd.DataFrame(data)
    
    if path.split('.')[-1] == "csv":
        if chunksize:
            chunks = pd.read_csv(path, chunksize=chunksize)
            # Initialize an empty list to store dataframes
            dfs = []
            for chunk in chunks:
                # Process each chunk (if needed). For demonstration, we're simply appending them to a list.
                dfs.append(chunk)
            # Concatenate chunks back into a single dataframe.
            df = pd.concat(dfs, axis=0)
            logger.info(f"Finished reading {path}")
        if logging:
            _check_memory_usage(path=path, chunksize=chunksize)
        return df