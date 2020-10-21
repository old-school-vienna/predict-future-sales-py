import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as kerasl

import df_data_struct as sdat
import df_data_flat as fdat
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


@dataclass
class LayerConfig:
    size_relative: float


@dataclass
class ModelConfig:
    activation: str
    optimizer: str
    loss: str
    layers: typing.List[LayerConfig]


def _create_model(model_config: ModelConfig, input_size: int):
    model = keras.Sequential()
    model.add(kerasl.Dense(input_size, activation=model_config.activation))
    for layer in model_config.layers:
        model.add(kerasl.Dense(int(layer.size_relative * input_size), activation=model_config.activation))
    model.add(kerasl.Dense(1))
    model.compile(optimizer=model_config.optimizer, loss=model_config.loss)
    return model


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


def mc(act: str, ls: typing.List[float]) -> ModelConfig:
    l1 = [LayerConfig(size) for size in ls]
    return ModelConfig(activation=act, optimizer='adam', loss='mean_squared_error', layers=l1)


def train():
    epochs = 40
    # int values smaller 32
    batch_sizes = [10]
    # list of lists of relative number of nodes for intermediate layers
    # [] .. No intermediate layer
    # [0.5] .. One intermediate layers with 50% of nodes of the input layer
    # Output layer has always one node
    layers_list = [[], [0.5], [0.6, 0.3], [0.7, 0.5, 0.2]]
    # Possible values: 'relu', 'sigmoid', 'tanh', 'softmax', ...
    activations = ["relu"]
    # Possible values fdat.read_train_data(), sdat.read_train_data()
    trainsets = [fdat.read_train_data()]

    for batch_size in batch_sizes:
        for layers in layers_list:
            for activation in activations:
                for ts in trainsets:
                    complexity = len(layers)
                    model_config = mc(activation, layers)
                    input_size = ts.x.shape[1]
                    model = DeepModel(f'{activation}_{complexity}', lambda: _create_model(model_config, input_size))
                    training = Training(id=f'bs{batch_size}', batch_size=batch_size,
                                        deepModel=model, epochs=epochs, trainset=ts)
                    plot_loss_during_training(training=training)


if __name__ == '__main__':
    train()
