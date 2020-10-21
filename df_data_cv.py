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
    deepModel: DeepModel


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


def plot_loss_during_training(training: Training, trainset: hlp.Trainset):
    plt.clf()
    pos = [1, 1, 1]
    plt.subplot(*pos)
    batch_sizes = [training.batch_size] * 5

    def fit_plot(batch_size: int):
        model = training.deepModel.create_lambda()
        hist = model.fit(trainset.x, trainset.y, epochs=4, batch_size=batch_size)
        plt.plot(hist.history['loss'])

    [fit_plot(e) for e in batch_sizes]
    # plt.legend([str(e) for e in batch_sizes], title='batch size')
    plt.yscale('log')
    plt.ylim(0.00001, 0.1)
    plt.title(f'training:{training.id} model:{training.deepModel.id} batchsize:{training.batch_size}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    fnam = hlp.dd() / f"train_{trainset.id}_{training.id}_{training.deepModel.id}.svg"
    plt.savefig(fnam, format='svg')
    print("---")
    print(f"- running training {training.id}")
    print(f"- with model {training.deepModel.id}")
    print(f"- on trainset {trainset.id}")
    print("--- saved to", fnam)


def mc(act: str, ls: typing.List[float]) -> ModelConfig:
    l1 = [LayerConfig(size) for size in ls]
    return ModelConfig(activation=act, optimizer='adam', loss='mean_squared_error', layers=l1)


def train(trainset: hlp.Trainset):
    input_size = trainset.x.shape[1]

    batch_sizes = [10, 20, 30]
    layers_list = [[], [0.5], [0.5, 0.3]]
    activations = ["sigmoid", "tanh", "relu"]

    for batch_size in batch_sizes:
        for layers in layers_list:
            for activation in activations:
                complexity = len(layers)
                model_config = mc(activation, layers)
                model = DeepModel(f'{activation}_{complexity}', lambda: _create_model(model_config, input_size))
                training = Training(id=f'bs{batch_size}', batch_size=batch_size, deepModel=model)
                plot_loss_during_training(training, trainset)


if __name__ == '__main__':
    ts = fdat.read_train_data()
    print("-- td x shape", ts.id, ts.x.shape)
    print("-- td y shape", ts.id, ts.y.shape)
    train(ts)
