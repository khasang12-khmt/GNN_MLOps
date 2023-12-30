import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train_opt
import mlflow
from prefect import flow

np.random.seed(555)


# parser = argparse.ArgumentParser()

# # movie
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''

class Params:
    def __init__(self, dataset, aggregator, n_epochs, neighbor_sample_size, dim, n_iter, batch_size, l2_weight, lr, ratio):
        self.dataset = dataset
        self.aggregator = aggregator
        self.n_epochs = n_epochs
        self.neighbor_sample_size = neighbor_sample_size
        self.dim = dim
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        self.lr = lr
        self.ratio = ratio
        
    def to_dict(self):
        return {
            "dataset": self.dataset,
            "aggregator": self.aggregator,
            "n_epochs": self.n_epochs,
            "neighbor_sample_size": self.neighbor_sample_size,
            "dim": self.dim,
            "n_iter": self.n_iter,
            "batch_size": self.batch_size,
            "l2_weight": self.l2_weight,
            "lr": self.lr,
            "ratio": self.ratio
        }

@flow
def train():
    mlflow.tensorflow.autolog()
    
    dataset = 'movie'
    aggregator = 'sum'
    n_epochs = 10
    neighbor_sample_size = 4
    dim = 32
    n_iter = 2
    batch_size = 65536
    l2_weight = 1e-7
    lr = 2e-2
    ratio = 1
    param = Params(dataset, aggregator, n_epochs, neighbor_sample_size, dim, n_iter, batch_size, l2_weight, lr, ratio)

    show_loss = False
    show_time = True
    show_topk = False

    t = time()

    data = load_data(param)
    train_opt(param, data, show_loss, show_topk)

    if show_time:
        print('time used: %d s' % (time() - t))
