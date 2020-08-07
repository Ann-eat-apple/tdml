## Tests

To run the tests from the Makefile, install the required packages first:
```shell
pip install -r requirements.txt
```
Run test for just a module and an ML framework:
```shell
make test_dataset_numpy
```
Run test for just a module in TDML:
```shell
make test_dataset
```
Run all the tests:
```shell
make test
```