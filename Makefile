.DEFAULT_GOAL := all

ifeq ($(PYTHON),)
	ifeq ($(shell which python3),)
		PYTHON = python
	else
		PYTHON = python3
	endif
endif

.PHONY: install
install:
	$(PYTHON) setup.py install

.PHONY: develop
develop:
	$(PYTHON) setup.py develop

.PHONY: all
all: install clean

.PHONY: dev
dev: develop clean

.PHONY: test_dataset_numpy
test_dataset_numpy:
	pytest tests/test_dataset/test_dataset_numpy.py --cov=tdml

.PHONY: test_dataset_pytorch
test_dataset_pytorch:
	pytest tests/test_dataset/test_dataset_pytorch.py --cov=tdml

.PHONY: test_dataset_tensorflow
test_dataset_tensorflow:
	pytest tests/test_dataset/test_dataset_tensorflow.py --cov=tdml

.PHONY: test_dataset
test_dataset: test_dataset_numpy test_dataset_pytorch test_dataset_tensorflow clean

.PHONY: test_data_numpy
test_data_numpy:
	pytest tests/test_data/test_data_numpy.py --cov=tdml

.PHONY: test_data
test_data: test_data_numpy clean

.PHONY: test
test: test_data test_dataset clean

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
