import argparse

import model
from dataset import get_dataloaders
from utils import Optimizer, Trainer, Recorder


def train(nb_epochs, repository, dataset_length, batch_size):
    # Dataset
    train_loader, valid_loader = get_dataloaders(repository, batch_size=batch_size, length=dataset_length)

    # Model
    # mdl = model.LinearModel(train_loader.input_size(), train_loader.num_classes())
    mdl = model.ConvModel(train_loader.input_size(), train_loader.num_classes())

    # Train parameters
    opt = Optimizer(mdl.parameters_and_gradients())
    rec = Recorder()

    trainer = Trainer(mdl, (train_loader, valid_loader), opt, rec)
    trainer.fit(nb_epochs)
    rec.plot()


def main():
    parser = argparse.ArgumentParser(description='Train MNIST dataset only with Numpy')
    parser.add_argument('nb_epochs', type=int, default=10, help='Number of epochs in training')
    parser.add_argument('repository', type=str, help='Where you want MNIST dataset to be downloaded')
    parser.add_argument('dataset_length', type=int, default=1000, help='Number of samples in training dataset')
    parser.add_argument('batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()