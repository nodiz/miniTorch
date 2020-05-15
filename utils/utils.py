import torch
from time import time
from datetime import timedelta
from math import sqrt, pi


def mean(lst):
    return sum(lst)/len(lst)


def get_accuracy(pred, target):
    assert len(pred) == len(target)
    return sum([p == t for (p,t) in zip(pred, target)])/len(target)


def demoDataset(nb_elements, batch_size):
    nb_elements -= nb_elements % batch_size

    data = torch.empty(nb_elements//batch_size, batch_size, 2).uniform_(0,1)
    targets = data.sub(0.5).pow(2).sum(2) < 1/(2*sqrt(pi))
    labels = torch.cat((targets, ~ targets)).view(2,-1).t().type(torch.double)
    targets = (~targets).type(torch.double)

    return data, targets, labels


def train(model, criterion, n_epochs, data, labels, lr, verbose=True):
    model.train()
    loss_list = list()

    n_batch, len_batch = data.shape[0], data.shape[1]
    y = labels.view(n_batch, len_batch,-1)

    # Train
    start_time = time()
    for e in range(n_epochs):
        for batch_i, (batch_x, batch_y)  in enumerate(zip(data, y)):

            output = model(batch_x)
            loss, dloss = criterion(output, batch_y)
            model.backward(dloss)
            model.optim_sgd_step(lr)
            model.zero_grad()

            loss_list.append(loss)
            time_left = timedelta(seconds=n_epochs-e * (time() - start_time) / (e + 1))
            log_str = f"--- Epoch {e}/{n_epochs} batch {batch_i}/{len(data)} ---\n" \
                      f"--- Loss {loss}\n ---" \
                      f"--- ETA {time_left}s ---\n"
            if verbose:
                print(log_str)

    # Accuracy on train dataset
    acc = validate(model, criterion, n_epochs, data, labels)

    return acc, loss_list


def validate(model, criterion, n_epochs, data, labels):
    model.eval()
    acc_list = list()

    n_batch, len_batch = data.shape[0], data.shape[1]
    targets = labels.argmax(1).view(n_batch, len_batch)

    for batch_i, (x, y) in enumerate(zip(data, targets)):
        output = model(x)
        y_pred = output.argmax(1)
        acc_list.append(get_accuracy(y_pred, y))

    return mean(acc_list)


