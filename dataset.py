import os
import urllib.request
import gzip
import pickle
import math
import numpy as np


MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'


class Dataset(object):
    def __init__(self, ds, length=None, type='train'):
        self.x = np.array(ds[0])
        self.y = np.array(ds[1])

        if length is not None:
            if type == 'valid':
                length = int(length * 0.3)

            self.x = self.x[:length]
            self.y = self.y[:length]

        self.size = int(math.sqrt(len(self.x[0])))

    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        img = img.reshape(self.size, self.size, 1)
        return img, label

    def __len__(self):
        return len(self.y)

    def mean(self):
        return self.x.mean()

    def std(self):
        return self.x.std()

    def normalize(self, mean, std):
        self.x = (self.x - self.x.mean()) / self.x.std()

    def input_size(self):
        #return self.size * self.size
        return self.size, self.size, 1

    def num_classes(self):
        return self.y.max() + 1


class Sampler:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.n, self.bs, self.shuffle = len(dataset), batch_size, shuffle

    def __iter__(self):
        self.idxs = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i: i + self.bs]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.ds, self.bs, self.shuffle = dataset, batch_size, shuffle
        self.sampler = Sampler(dataset, batch_size, shuffle)

    def __iter__(self):
        for s in self.sampler: yield self.collate([self.ds[i] for i in s])

    def __len__(self):
        return len(self.ds) // self.bs

    def collate(self, b):
        xs, ys = zip(*b)
        return np.stack(xs), np.stack(ys)

    def num_classes(self):
        return self.ds.num_classes()

    def input_size(self):
        return self.ds.input_size()


def get_dataset(repository, length=None):
    train_data, valid_data, _ = load_mnist(repository)
    train_ds, valid_ds = Dataset(train_data, length=length, type='train'), Dataset(valid_data, length=length, type='valid')

    train_mean = train_ds.mean()
    train_std = train_ds.std()
    train_ds.normalize(train_mean, train_std)
    valid_ds.normalize(train_mean, train_std)

    return train_ds, valid_ds


def get_dataloaders(repository, batch_size=1, length=None):
    train_ds, valid_ds = get_dataset(repository, length=length)
    return DataLoader(train_ds, batch_size, shuffle=True), DataLoader(train_ds, batch_size, shuffle=False)


def load_mnist(repository):
    mnist_path = download_mnist(repository)
    return extract_pickle(mnist_path)


def download_mnist(repository):
    mnist_path = os.path.join(repository, 'mnist.pkl.gz')

    if not os.path.isfile(mnist_path):
        urllib.request.urlretrieve(MNIST_URL, mnist_path)

    return mnist_path


def extract_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        p = pickle.load(f, encoding='latin-1')
        return p