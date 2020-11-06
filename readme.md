This is a Pytorch implementation of the paper: A Reinforcement Learning Based Strategy for Dynamic Scheduling on Heterogeneous Platforms.


## Requirements

  * Python 3.7
  * For the other packages, please refer to the requirements.txt.


## Usage

The main script is `train_new.py` . You can reproduce the results by:
```
pip install -r requirements.txt
python train_new.py
```
Some packages (PyTorch, torch-geometric) may have to be installed separately.

## Options

### DAG types
It is possible to train on three types of DAG: **Cholesky**, **QR**, **LU**, like in the following examples:

```
python train_new.py -env_type chol
python train_new.py -env_type QR
python train_new.py -env_type LU
```
### Number of tiles *T*

The parameter **n** controls the number of tiles *T*. As the graphs grow rapidly with the number of tiles, the training can become slow from 10 onwards.
```
python train_new.py -env_type QR -n 4
```

### Number of CPUs and GPUs

To change the number of processors on which the scheduling is made:
```
python train_new.py -nCPU 2 -nGPU 2
```

### Number of CPUs and GPUs

For other hyper-parameters concerning the RL algorithm or training, please look at the parser arguments at the top of the train_new file.