# Transform Dataframe for Machine Learning

[![PyPI](https://img.shields.io/pypi/v/tdml.svg)](https://pypi.org/project/tdml/) ![version](https://img.shields.io/badge/version-0.1.1-blue) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zechengz/tdml/blob/master/LICENSE) ![Build Status](https://travis-ci.com/zechengz/tdml.svg)

A lightweight and easy-to-use Python package that transforming dataframes into machine learning friendly data format.

<p align="center">
  <img width="406" height="150" src="https://github.com/zechengz/tdml/blob/master/docs/source/_img/tdml.png">
</p>

Current dataframes, including Pandas and PySpark, are widely used to manipulate tabular data. These packages provide rich functionalities and optimizations for data processing. But after the processing, data is usually input to machine learning or deep learning models, which are constructed by other ML packages. In this step, the user needs to spend time in transforming dataframes into arrays or tensors, splitting data into several sets, mapping categorical data to integers, and even representing text data by vectors. To make the whole process more efficient, TDML bridges the dataframes and ML frameworks by addressing the aforementioned painful issues. Currently, TDML provides functions including:

* Automatically transform dataframe (Pandas or PySpark) into ML framework (NumPy, PyTorch or TensorFlow) arrays or tensors.
* Map categorical data to integers, represent text data by bag-of-word and support UDF on text transformation.
* Split transformed data into several sets (train-test or train-validation-test) by one line of code.
* Support reshuffling (with random seed) the train set after the splitting.

## Examples
```shell
cd examples/numpy
python simple_sklearn_regression.py
```
For more examples, please go to the [examples](https://github.com/zechengz/tdml/tree/master/examples) folder.

## Tests
Please refer to the [tests](https://github.com/zechengz/tdml/tree/master/tests).

## Contact
[zecheng@cs.stanford.edu](mailto:zecheng@cs.stanford.edu)
