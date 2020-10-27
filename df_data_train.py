import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt

import df_data_flat as fdat
import df_data_next as ndat
import helpers as hlp

"""
flat: 891, 400
struct: 63, 40
model.compile(optimizer="adam", loss="mean_squared_error")
"""


@dataclass
class DeepModel:
    id: str
    create_lambda: typing.Callable


@dataclass
class Training:
    id: str
    batch_size: int
    epochs: int
    deepModel: DeepModel
    trainset: hlp.Trainset


def plot_loss_during_training(training: Training):
    plt.clf()
    pos = [1, 1, 1]
    plt.subplot(*pos)
    batch_sizes = [training.batch_size] * 5

    def fit_plot(batch_size: int):
        model = training.deepModel.create_lambda()
        hist = model.fit(training.trainset.x, training.trainset.y, epochs=training.epochs, batch_size=batch_size)
        plt.plot(hist.history['loss'])

    [fit_plot(e) for e in batch_sizes]
    # plt.legend([str(e) for e in batch_sizes], title='batch size')
    plt.yscale('log')
    plt.ylim(0.00001, 0.1)
    plt.title(f'trainset:{training.trainset.id} training:{training.id} model:{training.deepModel.id}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(0, training.epochs + 1, int(float(training.epochs + 1) / 10.0)))
    fnam = hlp.dd() / f"train_{training.trainset.id}_{training.id}_{training.deepModel.id}.svg"
    plt.savefig(fnam, format='svg')
    print("---")
    print(f"- running training {training.id}")
    print(f"- with model {training.deepModel.id}")
    print(f"- on trainset {training.trainset.id}")
    print("--- saved to", fnam)


def train():
    epochs = 40
    # int values smaller 32
    batch_sizes = [10, 5]
    # list of lists of relative number of nodes for intermediate layers
    # [] .. No intermediate layer
    # [0.5] .. One intermediate layers with 50% of nodes of the input layer
    # Output layer has always one node
    layers_list = [[1.0], [1.0, 1.0, 1.0],  [1.0, 1.0, 1.0, 1.0, 1.0] ]
    # Possible values: 'relu', 'sigmoid', 'tanh', 'softmax', ...
    activations = ["relu"]
    # Possible values fdat.read_train_data(), sdat.read_train_data()
    trainsets = [ndat.read_train_data()]

    for batch_size in batch_sizes:
        for layers in layers_list:
            for activation in activations:
                for ts in trainsets:
                    complexity = len(layers)
                    layer_configs = [hlp.LayerConfig(size) for size in layers]
                    model_config = hlp.ModelConfig(activation=activation, optimizer='adam',
                                                   loss='mean_squared_error', layers=layer_configs)
                    input_size = ts.x.shape[1]
                    model = DeepModel(f'{activation}_{complexity}', lambda: hlp.create_model(model_config, input_size))
                    training = Training(id=f'bs{batch_size}', batch_size=batch_size,
                                        deepModel=model, epochs=epochs, trainset=ts)
                    plot_loss_during_training(training=training)


if __name__ == '__main__':
    train()
