import matplotlib.pyplot as plt
from torch import FloatTensor
from functional.ReLu import ReLu
from functional.Tanh import Tanh
from layer.Linear import Linear
from loss.LossMSE import LossMse
from module.Sequential import Sequential
from optim.Optim import BengioSGD
from utils.utils import trainAndVal, Result


def mainCrossVal(model):
    criterion = LossMse()
    n_epochs = 100
    crossVal = list()
    for lr in [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07]:
        model.init()
        optim = BengioSGD(model.parameters(), lr=lr, momentum=0.9)
        acc_train, acc_val, losses = trainAndVal(model, optim, criterion, n_epochs, requires_plot=False)
        crossVal.append(Result(acc_train, acc_val, losses, lr))

    lrs = [result.lr for result in crossVal]
    plt.figure(0)
    for result in crossVal:
        plt.plot(result.losses)
    plt.title('Losses for lr')
    plt.legend(lrs)
    plt.show()


def mainStdAcc(model, lr):
    optim = BengioSGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = LossMse()
    n_epochs = 100
    crossVal = list()

    for trials in range(10):
        model.init()
        optim = BengioSGD(model.parameters(), lr=lr, momentum=0.9)
        acc_train, acc_val, losses = trainAndVal(model, optim, criterion, n_epochs, requires_plot=False)
        crossVal.append(Result(acc_train, acc_val, losses, lr))

    ts = [result.acc_train for result in crossVal]
    vs = [result.acc_val for result in crossVal]
    train_accs = FloatTensor(ts)
    val_accs = FloatTensor(vs)

    print(f"Type\tAccuracy\tstd\n"
          f"Train\t{train_accs.mean()}\t{train_accs.std()}\n"
          f"Val\t{val_accs.mean()}\t{val_accs.std()}\n")


def mainDemo(model, lr):
    optim = BengioSGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = LossMse()
    n_epochs = 100
    trainAndVal(model, optim, criterion, n_epochs, requires_plot=True, verbose=True)


if __name__ == "__main__":
    model = Sequential(('fc1', Linear(2, 25, use_bias=True)), ('relu1', ReLu()),
                       ('fc2', Linear(25, 25, use_bias=True)), ('relu2', ReLu()),
                       ('fc3', Linear(25, 25, use_bias=True)), ('relu3', ReLu()),
                       ('fc4', Linear(25, 2, use_bias=True, activ='tanh')), ('tanh1', Tanh()))

    print("Running lr crossvalidation")
    mainCrossVal(model)
    print("Calculating std for the best lr")
    mainStdAcc(model, lr=0.0001)
    print("Demoing best lr")
    mainDemo(model, lr=0.0001)
