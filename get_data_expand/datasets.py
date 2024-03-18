import os


import random
import tarfile
from urllib.request import urlopen, urlretrieve

import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

# 其实我认为这个是关键，也就是write_output函数，这个函数的作用是将数据写入到hdf5文件中
def write_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, point_type: str = "float", count: int = 100000) -> None:# 我建议直接10w个，我看可能需要多长时间
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.（fn）
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    from bruteforce.module import BruteForceBLAS

    with h5py.File(fn, "w") as f:
        print(f"writing {fn}...")
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 10 == 0:
                print(f"{i}/{len(test)}...")

            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


"""
param: train and test are arrays of arrays of indices.
"""




'''
def glove(out_fn: str, d: int) -> None:
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")
'''



'''
DATASETS: Dict[str, Callable[[str], None]] = {
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "sift-128-euclidean": sift,

}

`Dict[str, Callable[[str], None]]` 是 Python 类型注释（type hint），用于指示一个字典的类型。

在这个类型注释中：

- `Dict` 表示这个变量是一个字典。
- `[str, Callable[[str], None]]` 表示字典的键和值的类型。
  - `str` 表示字典的键是字符串类型。
  - `Callable[[str], None]` 表示字典的值是一个可调用的对象（函数或类），它接受一个字符串参数并返回空值 (`None`)。

因此，`Dict[str, Callable[[str], None]]` 表示这个字典的键是字符串类型，值是一个接受一个字符串参数并返回空值的可调用对象。


要使用 DATASETS 这个字典，您可以通过键来访问相应的值，这些值通常是函数。然后，您可以像调用普通函数一样调用这些值。
DATASETS["glove-25-angular"](output_file_name)

'''