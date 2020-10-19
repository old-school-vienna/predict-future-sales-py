import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow.python.keras as keras
import tensorflow.python.keras.activations as kerasa
import tensorflow.python.keras.layers as kerasl

import df_data_flat as dfd
import helpers as hlp

td = dfd.read_train_data()
print("-- td x shape", td.x.shape)
print("-- td y shape", td.y.shape)


@dataclass
class DeepModel:
    id: str
    desc: str
    create_lambda: typing.Callable


@dataclass
class Training:
    id: str
    desc: str
    batch_size: int
    deepModel: DeepModel


def create_model_one_inter_relu():
    return _create_model_one_inter(kerasa.relu)


def create_model_one_inter_sigmoid():
    return _create_model_one_inter(kerasa.sigmoid)


def create_model_one_inter_tanh():
    return _create_model_one_inter(kerasa.tanh)


def create_model_no_inter_relu():
    return _create_model_one_inter(kerasa.relu)


def create_model_no_inter_sigmoid():
    return _create_model_no_inter(kerasa.sigmoid)


def create_model_no_inter_tanh():
    return _create_model_no_inter(kerasa.tanh)


def _create_model_one_inter(activation: str):
    model = keras.Sequential()
    model.add(kerasl.Dense(891, activation=activation))
    model.add(kerasl.Dense(400, activation=activation))
    model.add(kerasl.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _create_model_no_inter(activation: str):
    model = keras.Sequential()
    model.add(kerasl.Dense(891, activation=activation))
    model.add(kerasl.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def plot_loss_during_training(training: Training):
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
    print(f"- running training {training.id}: {training.desc}")
    print(f"- with model {training.deepModel.id}: {training.deepModel.desc}")
    print("--- saved to", fnam)


def train():
    sm0 = DeepModel('sig0', 'No intermediate layer with sigmoid activation', create_model_no_inter_sigmoid)
    rm0 = DeepModel('relu0', 'No intermediate layer with relu activation', create_model_no_inter_relu)
    tm0 = DeepModel('tanh0', 'No intermediate layer with tanh activation', create_model_no_inter_tanh)

    sm1 = DeepModel('sig1', 'One intermediate layer with sigmoid activation', create_model_one_inter_sigmoid)
    rm1 = DeepModel('relu1', 'One intermediate layer with relu activation', create_model_one_inter_relu)
    tm1 = DeepModel('tanh1', 'One intermediate layer with tanh activation', create_model_one_inter_tanh)

    bss = [10, 20, 30]

    st0 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=sm0) for bs in bss]
    rt0 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=rm0) for bs in bss]
    tt0 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=tm0) for bs in bss]

    st1 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=sm1) for bs in bss]
    rt1 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=rm1) for bs in bss]
    tt1 = [Training(id=f'bs{bs}', desc=f'Batchsize {bs}', batch_size=bs, deepModel=tm1) for bs in bss]

    trainings = st1 + rt1 + tt1 + st0 + rt0 + tt0
    for t in trainings:
        print(t)
    for t in trainings:
        plot_loss_during_training(t)


if __name__ == '__main__':
    train()