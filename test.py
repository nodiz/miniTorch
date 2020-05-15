import torch
from utils.utils import demoDataset, train, validate, demoPlot
from functional.ReLu import ReLu, LeakyReLu
from functional.Tanh import Tanh
from layer.Linear import Linear
from loss.LossMSE import LossMse
from module.Sequential import Sequential

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    model = Sequential( ('fc1', Linear(2, 25, use_bias=True)), ('relu1', ReLu()),
                        ('fc2', Linear(25, 25, use_bias=True)), ('relu2', ReLu()),
                        ('fc3', Linear(25, 25, use_bias=True)), ('relu3', ReLu()),
                        ('fc4', Linear(25, 2, use_bias=True, activ='tanh')), ('tanh1', Tanh()))

    criterion = LossMse()

    n_epochs = 500
    lr = 0.001

    batch_size = 50
    trainX, _, trainY = demoDataset(1000, batch_size)
    valX, _, valY = demoDataset(1000, batch_size)

    print(f"Training with lr = {lr} over {n_epochs} epochs")

    acc_train, losses = train(
        model=model,
        criterion=criterion,
        data=trainX,
        labels=trainY,
        n_epochs=n_epochs,
        lr=lr
    )

    acc_val = validate(
        model=model,
        criterion=criterion,
        data=valX,
        labels=valY,
        n_epochs=n_epochs,
    )

    print(f"Completed training and validation:\n"
          f"Training accuracy: {acc_train}\n"
          f"Validation accuracy: {acc_val}\n"
          f"Loss history {losses}")

    sns.set()
    plt.figure(0)
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.show()

    demoPlot(model, data=valX)
