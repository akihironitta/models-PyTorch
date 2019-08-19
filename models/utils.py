import argparse
import os
import torch
import torchvision
from torchvision import transforms


def parse_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_size', type=int, default=784)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--out_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='mnist')
    
    return parser.parse_args()


def load_dataloader(dataset, batch_size, datadir="/tmp/data/"):
    """https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist"""

    if dataset == "mnist":
        # MNIST: http://yann.lecun.com/exdb/mnist/
        D = 784
        K = 10
        ds_train = torchvision.datasets.MNIST(root=datadir,
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)
        ds_test = torchvision.datasets.MNIST(root=datadir,
                                             train=False,
                                             transform=transforms.ToTensor(),
                                             download=True)
    elif dataset == "emnist":
        # EMNIST: https://www.nist.gov/itl/iad/image-group/emnist-dataset/
        D = 784
        K = 10
        ds_train = torchvision.datasets.EMNIST(root=datadir,
                                               split="letters",
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
        ds_test = torchvision.datasets.EMNIST(root=datadir,
                                              split="letters",
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    elif dataset == "fmnist":
        # Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
        D = 784
        K = 10
        ds_train = torchvision.datasets.FashionMNIST(root=datadir,
                                                     train=True,
                                                     transform=transforms.ToTensor(),
                                                     download=True)
        ds_test = torchvision.datasets.FashionMNIST(root=datadir,
                                                    train=False,
                                                    transform=transforms.ToTensor(),
                                                    download=True)
    elif dataset == "cifar10":
        # CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
        D = 32*32*3
        K = 10
        ds_train = torchvision.datasets.CIFAR10(root=datadir,
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
        ds_test = torchvision.datasets.CIFAR10(root=datadir,
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    elif dataset == "cifar100":
        # CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html
        D = 32*32*3
        K = 100
        ds_train = torchvision.datasets.CIFAR100(root=datadir,
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
        ds_test = torchvision.datasets.CIFAR100(root=datadir,
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)

    elif dataset == "svhn":
        # SVHN: http://ufldl.stanford.edu/housenumbers/
        D = 32*32*3
        K = 10
        ds_train = torchvision.datasets.SVHN(root=datadir,
                                             split="train",
                                             transform=transforms.ToTensor(),
                                             download=True)
        ds_test = torchvision.datasets.SVHN(root=datadir,
                                            split="test",
                                            transform=transforms.ToTensor(),
                                            download=True)
    else:
        assert False, "The dataset \"" + dataset + "\" is not supported."

    dl_train = torch.utils.data.DataLoader(dataset=ds_train,
                                           batch_size=batch_size,
                                           shuffle=True)
    dl_test = torch.utils.data.DataLoader(dataset=ds_test,
                                           batch_size=batch_size)
    return dl_train, dl_test


def save_model(model, idx=0, dataset_name="mnist", state_dir="./checkpoints", add_name=""):
    state_filename = model.name + '-' + dataset_name + '-' + str(idx) + add_name + ".pkl"
    state_path = os.path.join(state_dir, state_filename)
    torch.save(model.state_dict(), state_path)
    

def load_model(model, idx=0, dataset_name="mnist", state_dir="./checkpoints", add_name=""):
    state_filename = model.name + '-' + dataset_name + "-" + str(idx) + add_name + ".pkl"
    state_path = os.path.join(state_dir, state_filename)
    model.load_state_dict(torch.load(state_path))


def embed_into_2d(X, method="tsne"):
    """
    Data points are embedded into two dimensional space
    if they are in more than two dimensional space.
    
    X --- numpy array (N, D)
    Y --- numpy array (N,)
    """
    supported_methods = {"tsne", "umap"}
    if method not in supported_methods:
        pass
    
    if X.shape[1] == 2:
        Z = X
    else:
        if method == "tsne":
            from sklearn.manifold import TSNE
            print("Embedding X into 2 dimensional space with t-SNE...")
            Z = TSNE(n_components=2).fit_transform(X)
        elif method == "umap":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    return Z


def test_embed_into_2D():
    import matplotlib.pyplot as plt
    import numpy as np
    D = 100
    N = 1000
    method = "tsne"

    # data points
    X1 = np.random.randn(N//2, D)
    X2 = np.random.randn(N//2, D) + 1
    X = np.concatenate((X1, X2))

    # data labels
    Y1 = np.zeros(N//2)
    Y2 = np.ones(N//2)
    Y = np.concatenate((Y1, Y2))

    # embed data if needed
    Z = embed_into_2d(X, method)

    # plot the (embedded) data
    plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="gist_rainbow", marker=".")
    plt.show()

if __name__ == '__main__':
    test_embed_into_2D()
    # print(parse_parameters())
    # print(load_dataloader("cifar10", 256))
