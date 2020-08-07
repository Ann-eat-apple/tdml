Background
==========

With the development of data manipulation and machine learning, 
several packages are widely used in Python data analysis. 
The packages included in TDML and the reasons why they are included 
are listed below.

.. contents::
    :local:

Dataframes
##########

TDML currently supports two dataframes. One is `Pandas <https://pandas.pydata.org/>`_ and another one is 
`PySpark <https://spark.apache.org/docs/latest/api/python/index.html>`_.

In general, `Pandas <https://pandas.pydata.org/>`_ provides rich and clear functionalities to manipulate 
the tabular data. `PySpark <https://spark.apache.org/docs/latest/api/python/index.html>`_, on the other hand, 
can connect to database, has a lot of optimizations for performance and is able to be used on big data applications.

ML Frameworks
#############

TDML supports three machine learning frameworks: `NumPy <https://numpy.org/>`_, 
`PyTorch <https://pytorch.org/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.

For simple ML models, people usually use `NumPy <https://numpy.org/>`_ with `scikit-learn <https://scikit-learn.org>`_.
But if more complex models are required, usually for deep learning methods, `PyTorch <https://pytorch.org/>`_ and 
`TensorFlow <https://www.tensorflow.org/>`_ are mostly used.