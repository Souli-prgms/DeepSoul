import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

import functional


class Optimizer:
    def __init__(self, params_and_grads, lr=0.001, weight_decay=1e-4, epsilon=1e-8):
        self.params_and_grads, self.lr, self.weight_decay, self.epsilon = list(params_and_grads), lr, weight_decay, epsilon
        self.momentum_cache, self.rmsprop_cache = [], []
        self.init_caches()

    def step(self):
        self.momentum()
        self.rmsprop()

        for i, (p, grad) in enumerate(self.params_and_grads):
            for key, value in p.items():
                first_moment = self.momentum_cache[i][key]
                second_moment = self.rmsprop_cache[i][key]

                learning_rate = self.lr / (np.sqrt(second_moment) + 1e-8)

                value -= learning_rate * (first_moment + self.weight_decay * grad[key])

    def zero_grad(self):
        for _, grad in self.params_and_grads:
            for key, value in grad.items():
                value.fill(0)

    def init_caches(self):
        for l in self.params_and_grads:
            new_grads = {}
            for key, value in l[0].items():
                new_grads[key] = np.zeros(value.shape)
            self.momentum_cache.append(copy.deepcopy(new_grads))
            self.rmsprop_cache.append(copy.deepcopy(new_grads))

    def momentum(self, beta=0.9):
        for i, (_, grad) in enumerate(self.params_and_grads):
            for key, value in grad.items():
                self.momentum_cache[i][key] = beta * self.momentum_cache[i][key] + (1 - beta) * value

    def rmsprop(self, beta=0.999):
        for i, (_, grad) in enumerate(self.params_and_grads):
            for key, value in grad.items():
                new_grad = beta * self.rmsprop_cache[i][key] + (1 - beta) * (value ** 2)
                self.rmsprop_cache[i][key] = np.maximum(new_grad, self.rmsprop_cache[i][key])


class Trainer:
    def __init__(self, model, ds_loaders, optimizer, recorder):
        self.model = model
        self.train_loader = ds_loaders[0]
        self.valid_loader = ds_loaders[1]
        self.optimizer = optimizer
        self.recorder = recorder

    def fit(self, nb_epochs):
        for i in range(nb_epochs):
            print("Epoch {}".format(i))
            self.train_step()
            self.valid_step()
            self.recorder.reset_for_next_epoch()

    def train_step(self):
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            pred, loss = self.model(inputs, to_one_hot(targets, self.train_loader.num_classes()))
            self.model.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.recorder.train_step(1, targets, pred, loss)

    def valid_step(self):
        for i, (inputs, targets) in enumerate(tqdm(self.valid_loader)):
            pred, loss = self.model(inputs, to_one_hot(targets, self.valid_loader.num_classes()))
            self.recorder.valid_step(1, targets, pred, loss)


class Recorder:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        self.current_n = {'train': 0.0, 'valid': 0.0}
        self.current_tot_loss = {'train': 0.0, 'valid': 0.0}
        self.current_correct = {'train': 0.0, 'valid': 0.0}

    def train_step(self, n, targets, outputs, loss):
        self.current_n['train'] += n
        self.current_tot_loss['train'] += loss
        self.current_correct['train'] += self.compute_accuracy(targets, outputs)

    def valid_step(self, n, targets, outputs, loss):
        self.current_n['valid'] += n
        self.current_tot_loss['valid'] += loss
        self.current_correct['valid'] += self.compute_accuracy(targets, outputs)

    def compute_accuracy(self, targets, outputs):
        return functional.accuracy(outputs, targets)

    def reset_for_next_epoch(self):
        if len(self.epochs) == 0:
            self.epochs.append(0)

        else:
            self.epochs.append(self.epochs[-1] + 1)

        train_loss = self.current_tot_loss['train'] / self.current_n['train']
        valid_loss = self.current_tot_loss['valid'] / self.current_n['valid']

        train_acc = self.current_correct['train'] / self.current_n['train']
        valid_acc = self.current_correct['valid'] / self.current_n['valid']

        print("\n Train : Loss : {:.4f}, Acc : {:.4f}".format(train_loss, train_acc))
        print(" Validation : Loss : {:.4f}, Acc : {:.4f} \n".format(valid_loss, valid_acc))

        self.train_loss.append(train_loss)
        self.val_loss.append(valid_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(valid_acc)

        self.reset()

    def reset(self):
        self.current_n = {'train': 0.0, 'valid': 0.0}
        self.current_tot_loss = {'train': 0.0, 'valid': 0.0}
        self.current_correct = {'train': 0.0, 'valid': 0.0}

    def plot(self):
        plt.plot(self.epochs, self.train_loss, 'r', label="Train loss")
        plt.plot(self.epochs, self.train_acc, 'b', label="Train acc")
        plt.plot(self.epochs, self.val_loss, 'g', label="Val loss")
        plt.plot(self.epochs, self.val_acc, 'y', label="Val acc")
        plt.legend()
        plt.show()


def to_one_hot(target, num_classes):
    bs = target.shape[0]
    one_hot = np.zeros((bs, num_classes))
    for i, val in enumerate(target):
        one_hot[i][val] = 1.0
    return one_hot