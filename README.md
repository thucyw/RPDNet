# RDNet

RDNet is the model that can detect real targets from RDM. For detail, see our [paper]().

## Prerequisites

- Python 3.6
- PyTorch 1.8+
- GPU


## Getting started
### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/download/)
- Install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive)
- install [cuDNN8.0](https://developer.nvidia.com/cudnn)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.8](http://pytorch.org/)

### Dataset
Download [Dataset]().

Before training or test, please make sure you have prepared the dataset
by organizing the directory as:
`data/your_dataset/data` and  `data/your_dataset/label`.
E.g. `data/GT6/data` and `data/GT6/label`.

### Configuration 

In `config/base_confige.yml`, you might want to change the following settings:
- `data` **(NECESSARY)** root path of the dataset for training or testing
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs
- `learning_rate` learning rate
- `batch_size` batch size for traning

### Train
Train a model by
```bash
python main.py train
```
- `--config` path of configuration file #Default: `config/base_config.yml`

### Evaluation
Evaluate the trained model by
```bash
python main.py eval
```
- `--config` path of configuration file #Default: `config/base_config.yml`
- `--epoch` iteration of the checkpoint to load. #Default: -1

It will output Precision rate, Recall rate, and number of target points

### Transform

Transform the prepared dataset using the trained model by

```bash
python main.py transform
```

- `--config` path of configuration file #Default: `config/base_config.yml`
- `--epoch` iteration of the checkpoint to load. #Default: -1


## License
RDNetis freely available for free non-commercial use, and may be redistributed under these conditions.
For commercial queries, contact [xxx]().