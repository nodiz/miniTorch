from datetime import timedelta
from math import sqrt, pi
from time import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch


def mean(lst):
    return sum(lst) / len(lst)


def xavier_init(in_size, out_size):
    return torch.empty((in_size, out_size)).uniform_(-1, 1) * sqrt(6. / (in_size + out_size))


def kaiming_init(in_size, out_size):
    return torch.randn((in_size, out_size)) * sqrt(2. / in_size)


def get_accuracy(pred: torch.Tensor, target: torch.Tensor):
    assert len(pred) == len(target)
    return pred.eq(target).sum().item() / len(target)


def demoDataset(nb_elements, batch_size):
    nb_elements -= nb_elements % batch_size

    data = torch.empty(nb_elements // batch_size, batch_size, 2).uniform_(0, 1)
    targets = data.sub(0.5).pow(2).sum(2).sqrt() < 1 / sqrt(2 * pi)
    labels = torch.cat((targets, ~targets)).type(torch.int32).view(2, -1).t()
    targets = (~targets).type(torch.int32)

    return data, targets, labels


def train(model, criterion, n_epochs, data, labels, optim, verbose=False):
    model.train()
    loss_list = list()

    n_batch, len_batch = data.shape[0], data.shape[1]
    y = labels.view(n_batch, len_batch, -1)

    # Train
    start_time = time()
    for e in range(n_epochs):
        epoch_loss = 0
        for batch_i, (batch_x, batch_y) in enumerate(zip(data, y)):
            optim.zero_grad()
            output = model(batch_x)
            loss, dloss = criterion(output, batch_y)
            model.backward(dloss)
            optim.step()

            epoch_loss += loss

        time_left = timedelta(seconds=n_epochs - e * (time() - start_time) / (e + 1))

        log_str = f"--- Epoch {e}/{n_epochs}  ---\n" \
                  f"--- Loss {epoch_loss}\n ---" \
                  f"--- ETA {time_left}s ---\n"
        if verbose:
            print(log_str)

        loss_list.append(epoch_loss)
    # Accuracy on train dataset
    acc = validate(model, criterion, n_epochs, data, labels)

    return acc, loss_list


def validate(model, criterion, n_epochs, data, labels, verbose=False):
    model.eval()
    acc_list = list()

    n_batch, len_batch = data.shape[0], data.shape[1]
    targets = labels.argmax(1).view(n_batch, len_batch)

    for batch_i, (x, y) in enumerate(zip(data, targets)):
        output = model(x)
        y_pred = output.argmax(1)
        acc_list.append(get_accuracy(y_pred, y))

    return mean(acc_list)


def trainAndVal(model, optim, criterion, n_epochs, requires_plot=False, verbose=False):
    batch_size = 50
    trainX, _, trainY = demoDataset(1000, batch_size)
    valX, _, valY = demoDataset(1000, batch_size)

    acc_train, losses = train(
        model=model,
        criterion=criterion,
        data=trainX,
        labels=trainY,
        n_epochs=n_epochs,
        optim=optim
    )

    acc_val = validate(
        model=model,
        criterion=criterion,
        data=valX,
        labels=valY,
        n_epochs=n_epochs,
    )

    if verbose:
        print(f"Completed training and validation:\n"
              f"Training accuracy: {acc_train}\n"
              f"Validation accuracy: {acc_val}\n")

    if requires_plot:
        sns.set()
        lossPlot(losses)
        demoPlot(model, data=valX)

    return acc_train, acc_val, losses


def demoPlot(model, data):
    model.eval()
    n_batch, len_batch = data.shape[0], data.shape[1]
    y = torch.Tensor().int()
    for batch_i, x in enumerate(data):
        output = model(x)
        y_pred = output.argmax(1)
        y = torch.cat((y, y_pred.int()))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    circle = plt.Circle((0.5, 0.5), 1 / sqrt(2 * pi), color='k', fill=False)
    ax.add_artist(circle)
    flat_x = [item.item() for sublist in data[:, :, 0] for item in sublist]
    flat_y = [item.item() for sublist in data[:, :, 1] for item in sublist]
    colors = ['teal', 'darkorchid']
    plt.scatter(flat_x, flat_y, c=y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title("Classification Demo")
    plt.show()


def lossPlot(losses):
    plt.figure(0)
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.show()


class Result:
    def __init__(self, acc_train, acc_val, losses, lr=0):
        self.acc_train = acc_train
        self.acc_val = acc_val
        self.losses = losses
        self.lr = lr


