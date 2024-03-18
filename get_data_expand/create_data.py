import os
import h5py
import numpy  as np
from typing import Any, Callable, Dict, Tuple

def read_dataset(fn: str) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Reads the training and testing data from the provided HDF5 file.

    Args:
        filename (str): The name of the HDF5 file from which data should be read.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, str, int]: The training data, the testing 
            data, the distance metric, and the number of neighbors to compute.
    """
    with h5py.File(fn, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
    return train, test

def get_data(fr:str, fn: str, distance: str, point_type: str = "float", count: int = 100000):
    '''
    Args:
    fr(str): The name of the HDF5 file from which data should be read.
    filename (str): The name of the HDF5 file to which data should be written.（fn）
    distance_metric (str): The distance metric to use for computing nearest neighbors.
    point_type (str, optional): The type of the data points. Defaults to "float".
    neighbors_count (int, optional): The number of nearest neighbors to compute for 
        each point in the test set. Defaults to 10[]
    '''
    from datasets import write_output
    train, test = read_dataset(fr) 
    test = test[:1000]
    print(test.shape)
    write_output(train, test, fn, distance, point_type=point_type, count=count)
# 这里就取了1000个测试数据，如果需要更多的数据可以修改这个数字


if __name__ == "__main__":
    # 首先读取原始数据，这一部风数据是已经下载好的，放在data文件夹中
    filename='../data/glove-25-angular.hdf5'
    count=100000
    # 这一部分扩展数据存放的地址
    out_put_fn = f'../data_expand/glove-25-angular-{count}.hdf5'
    get_data(filename, out_put_fn, 'angular', 'float', count)
