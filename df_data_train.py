import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt

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


@dataclass
class TrainConfig:
    epochs: int
    # int values smaller 32
    batch_sizes: typing.List[int]
    # list of lists of relative number of nodes for intermediate layers
    # [] .. No intermediate layer
    # [0.5] .. One intermediate layers with 50% of nodes of the input layer
    # Output layer has always one node
    layers_list: typing.List[typing.List[float]]
    # Possible values: 'relu', 'sigmoid', 'tanh', 'softmax', ...
    activations: typing.List[str]
    # Possible values fdat.read_train_data, sdat.read_train_data
    trainsets: typing.List[typing.Callable[[], hlp.Trainset]]


configs = {
    'next01': TrainConfig(
        epochs=40,
        batch_sizes=[10, 5],
        layers_list=[[1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_all],
    ),
    'next02': TrainConfig(
        epochs=30,
        batch_sizes=[10],
        layers_list=[[1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_M],
    ),
    'tryout': TrainConfig(
        epochs=10,
        batch_sizes=[5],
        layers_list=[[1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_all]
    )
}


def train(train_id: str, train_config: TrainConfig):
    print("== running", train_id )
    for batch_size in train_config.batch_sizes:
        for layers in train_config.layers_list:
            for activation in train_config.activations:
                for creator in train_config.trainsets:
                    ts = creator()
                    print("-- Trainset", ts.id)
                    complexity = len(layers)
                    print("-- NN", complexity)
                    layer_configs = [hlp.LayerConfig(size) for size in layers]
                    model_config = hlp.ModelConfig(activation=activation, optimizer='adam',
                                                   loss='mean_squared_error', layers=layer_configs)
                    input_size = ts.x.shape[1]
                    model = DeepModel(f'{activation}_{complexity}', lambda: hlp.create_model(model_config, input_size))
                    training = Training(id=f'{train_id}_bs{batch_size}', batch_size=batch_size,
                                        deepModel=model, epochs=train_config.epochs, trainset=ts)
                    plot_loss_during_training(training=training)


if __name__ == '__main__':
    tid = 'next02'
    train(tid, train_config=configs[tid])
