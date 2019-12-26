# DeepSoul
CNN from scratch (Numpy only) with MNIST logits as dataset

## Prerequisites
```
python3
numpy
matplotlib
tqdm
```

## How to run
Clone the repo on your machine and run this command:
```
python train.py nb_epochs path/to/mnist/future/download nb_samples_in_training batch_size
```

## TO DO
- Batch norm
- GPU training
- Data augmentation (+ try to test mixup)
- Learning rate scheduler
- Better weights init
- Dropout
- Deeper network
- Installable library (setup.py file)