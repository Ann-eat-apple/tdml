Introduction
============

Here is a quick introduction for the use of TDML.

.. contents::
    :local:

Choose Dataframe
----------------
Choose the dataframe before importing the TDML. Without specifying the dataframe, the default 
dataframe is `Pandas <https://pandas.pydata.org/>`_. Another one is 
`PySpark <https://spark.apache.org/docs/latest/api/python/index.html>`_.

.. code-block:: python

	import os
	os.environ['TDML_DATAFRAME'] = 'pyspark'
	# choose the PySpark as the dataframe
	import tdml

User can also specify the dataframe through the terminal:

.. code-block:: shell

	export TDML_DATAFRAME="pyspark"

Choose ML Framework
-------------------
Choose the ML framework before importing the TDML. Without specifying the ML framework, the default 
ML framework is `NumPy <https://numpy.org/>`_. The other two are `PyTorch <https://pytorch.org/>`_ and 
`TensorFlow <https://www.tensorflow.org/>`_.

.. code-block:: python

	import os
	os.environ['MLLIB'] = 'pytorch'
	# choose the PyTorch as the ML framework
	# os.environ['MLLIB'] = 'tensorflow'
	import tdml

Similarly, you can also specify the framework through `export` in terminal.

Load into Dataset
-----------------
To use the TDML, the first step is to load the dataframe into the :class:`tdml.Dataset`.
User can specify the `label`, `feature` or even `text` when loading into the dataset.

.. code-block:: python

	>>> import tdml
	>>> import pandas as pd

	>>> cars = {'Brand': ['Audi', 'Toyota', 'Ford', 'Audi', 'Mercedes', 'Mercedes', 'Audi', 'Audi', 'Toyota', 'Audi'],
				'Price': [30000, 25000, 27000, 35000, 55000, 60000, 31000, 31000, 23000, 32000],
				'Evaluation': ['Good', 'OK', 'Bad', 'Good', 'Good', 'Good', 'OK', 'Good', 'OK', 'Good']
		   	   }
	>>> df = pd.DataFrame(cars, columns = ['Brand', 'Price', 'Evaluation'])
	>>> ds = tdml.Dataset(df, label='Brand', feature='Price')

Transform Dataset
-----------------
After loading the data into :class:`tdml.Dataset`, user can use the `transform` function to transfrom the 
dataset into ML friendly data format. If user specified `text` data, user can add their own transformation functions 
to transform the text data.

.. code-block:: python
	
	>>> ds.transform()
	>>> ds
	>>> Dataset(label=[10], feature=[10, 1], label_mapping=4)

After the transformation, the `label`, `feature` and `label_mapping` (the categorical data will be automatically transformed and a mapping dictionary will be available) are available in the dataset object:

.. code-block:: python

	>>> ds.label
	>>> [0 3 1 0 2 2 0 0 3 0]
	>>> ds.feature.shape
	>>> (10, 1)
	>>> ds.label_mapping
	>>> {'Audi': 0, 'Ford': 1, 'Mercedes': 2, 'Toyota': 3}

Split into Sets
---------------

After the transformation, user can choose to split the dataset into train and test sets (or train, validation and test sets).

.. code-block:: python

	>>> ds.train_test_split(0.7, 0.3) # split 70% into train set, 30% into test set
	>>> ds
	>>> Dataset(label=[10], feature=[10, 1], train_x=[7, 1], train_y=[7], test_x=[3, 1], test_y=[3], label_mapping=4)
	>>> ds.train_x.shape
	>>> (7, 1)
	>>> ds.test_x.shape
	>>> (3, 1)
	>>> ds.train_y
	>>> [1 3 2 0 3 0 0]
	>>> ds.test_y
	>>> [0 0 2]

Input to Model and Reshuffle
----------------------------

At last, user can input the data into the model with the choice of reshuffling the train data.

.. code-block:: python

	>>> ds.reshuffle(seed=100) # reshuffle the train data
	>>> from sklearn.linear_model import LogisticRegression # import a simple model
	>>> model = LogisticRegression(random_state=0)
	>>> model.fit(ds.train_x, ds.train_y) # fit on the train set
	>>> model.score(ds.test_x, ds.test_y) # test on the test set
	>>> 0.6666666666666666
