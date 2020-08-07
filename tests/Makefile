# tests for dataset
.PHONY: test_dataset_numpy
test_dataset_numpy:
	pytest test_dataset/test_dataset_numpy.py --cov=tdml

.PHONY: test_dataset_pytorch
test_dataset_pytorch:
	pytest test_dataset/test_dataset_pytorch.py --cov=tdml

.PHONY: test_dataset_tensorflow
test_dataset_tensorflow:
	pytest test_dataset/test_dataset_tensorflow.py --cov=tdml

.PHONY: test_dataset
test_dataset: test_dataset_numpy test_dataset_pytorch test_dataset_tensorflow clean

# tests for data
.PHONY: test_data_numpy
test_data_numpy:
	pytest test_data/test_data_numpy.py --cov=tdml

.PHONY: test_data_pytorch
test_data_pytorch:
	pytest test_data/test_data_pytorch.py --cov=tdml

.PHONY: test_data_tensorflow
test_data_tensorflow:
	pytest test_data/test_data_tensorflow.py --cov=tdml

.PHONY: test_data
test_data: test_data_numpy test_data_pytorch test_data_tensorflow clean

# tests for pyspark
.PHONY: test_pyspark_numpy
test_pyspark_numpy:
	pytest test_pyspark/test_pyspark_numpy.py --cov=tdml

.PHONY: test_pyspark_pytorch
test_pyspark_pytorch:
	pytest test_pyspark/test_pyspark_pytorch.py --cov=tdml

.PHONY: test_pyspark_tensorflow
test_pyspark_tensorflow:
	pytest test_pyspark/test_pyspark_tensorflow.py --cov=tdml

.PHONY: test_pyspark
test_pyspark: test_pyspark_numpy test_pyspark_pytorch test_pyspark_tensorflow clean

.PHONY: test
test: test_data test_dataset test_pyspark clean

.PHONY: clean
clean:
	rm -rf *.pyc
	rm -rf __pycache__/
	rm -rf .DS_Store
	rm -rf dist
	rm -rf build
	rm -rf tdml.egg-info/
	rm -rf .coverage*
	rm -rf .pytest_cache