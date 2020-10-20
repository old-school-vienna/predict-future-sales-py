import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow.python.keras as keras
import tensorflow.python.keras.activations as kerasa
import tensorflow.python.keras.layers as kerasl

import df_data_flat as dat
# import df_data_struct as dat
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
    for l in model_config.layers:
        model.add(kerasl.Dense(int(l.size_relative * input_size), activation=model_config.activation))
    model.add(kerasl.Dense(1))
    model.compile(optimizer=model_config.optimizer, loss=model_config.loss)
    return model


def plot_loss_during_training(training: Training, td: hlp.Trainset):
    plt.clf()
    pos = [1, 1, 1]
    plt.subplot(*pos)
    batch_sizes = [training.batch_size] * 5

    def fit_plot(batch_size: int):
        model = training.deepModel.create_lambda()
        hist = model.fit(td.x, td.y, epochs=8, batch_size=batch_size)
        plt.plot(hist.history['loss'])

    [fit_plot(e) for e in batch_sizes]
    # plt.legend([str(e) for e in batch_sizes], title='batch size')
    plt.yscale('log')
    plt.ylim(0.00001, 1)
    plt.title(f'training:{training.id} model:{training.deepModel.id} batchsize:{training.batch_size}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    fnam = hlp.dd() / f"train_{training.id}_{training.deepModel.id}.svg"
    plt.savefig(fnam, format='svg')
    print("---")
    print(f"- running training {training.id}")
    print(f"- with model {training.deepModel.id}")
    print("--- saved to", fnam)


def mc(act: str, ls: typing.List[float]) -> ModelConfig:
    l1 = [LayerConfig(size) for size in ls]
    return ModelConfig(activation=act, optimizer='adam', loss='mean_squared_error', layers=l1)


def train(td: hlp.Trainset):
    input_size = td.x.shape[1]

    batch_sizes = [10, 20, 30]
    layers_list = [[], [0.5], [0.5, 0.3]]
    activations = [kerasa.sigmoid, kerasa.tanh, kerasa.relu]

    for batch_size in batch_sizes:
        for layers in layers_list:
            for activation in activations:
                complexity = len(layers)
                model_config = mc(activation, layers)
                model = DeepModel(f'{activation}_{complexity}', lambda: _create_model(model_config, input_size))
                training = Training(id=f'bs{batch_size}', batch_size=batch_size, deepModel=model)
                plot_loss_during_training(training, td)


if __name__ == '__main__':
    trainset = dat.read_train_data()
    print("-- td x shape", trainset.x.shape)
    print("-- td y shape", trainset.y.shape)
    train(trainset)
