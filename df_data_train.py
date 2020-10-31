import typing
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt

import df_data_next as ndat
import helpers as hlp


@dataclass
class DeepModel:
    id: str
    create_lambda: typing.Callable


@dataclass
class Training:
    id: str
    batch_size: int
    epochs: int
    iterations: int
    deepModel: DeepModel
    trainset: hlp.Trainset
    ylim_min: float
    ylim_max: float
    yscale: str


@dataclass
class TrainConfig:
    epochs: int
    iterations: int
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
    ylim_min: float
    ylim_max: float
    yscale: str


configs = {
    'next02': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_s, ndat.read_train_data_m, ndat.read_train_data_l],
        ylim_min=0.00001,
        ylim_max=0.1,
        yscale='log'

    ),
    'nextkarl01': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_karl],
        ylim_min=0.00001,
        ylim_max=0.1,
        yscale='log'
    ),
    'nextkarl02': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_karl_not_norm],
        ylim_min=150,
        ylim_max=500,
        yscale='linear'
    ),
    'nextkarl02a': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[
            [],
            [1.0],
            [1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        activations=["relu"],
        trainsets=[ndat.read_train_data_karl_not_norm],
        ylim_min=150,
        ylim_max=500,
        yscale='linear'
    ),
    'tryout': TrainConfig(
        epochs=5,
        iterations=2,
        batch_sizes=[5],
        layers_list=[[1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_s],
        ylim_min=0.00001,
        ylim_max=0.1,
        yscale='log'
    )
}


def fnam(training: Training) -> str:
    return f"train_{training.id}_{training.trainset.id}_{training.deepModel.id}"


def plot_loss_during_training(training: Training, history_list: typing.List[typing.List[float]]):
    plt.clf()
    pos = [1, 1, 1]
    plt.subplot(*pos)

    [plt.plot(history) for history in history_list]
    # plt.legend([str(e) for e in batch_sizes], title='batch size')
    plt.yscale(training.yscale)
    plt.ylim(training.ylim_min, training.ylim_max)
    plt.title(f'trainset:{training.trainset.id} training:{training.id} model:{training.deepModel.id}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(0, training.epochs + 1, int(float(training.epochs + 1) / 10.0)))
    fn = fnam(training)
    fp = hlp.dd() / f"{fn}.svg"
    plt.savefig(fp, format='svg')
    print("--- ploted to", fp)


def train_histories(training: Training) -> list:
    def train() -> list:
        model = training.deepModel.create_lambda()
        hist = model.fit(training.trainset.x, training.trainset.y, epochs=training.epochs,
                         batch_size=training.batch_size)
        return list(hist.history['loss'])

    history_list = [train() for _ in range(training.iterations)]
    result = {
        'id': training.id,
        'model': training.deepModel.id,
        'data': training.trainset.id,
        'histories': history_list
    }
    fn = fnam(training)
    fp = hlp.dd() / f"{fn}.json"
    with open(fp, 'w') as file:
        json.dump(result, file)
        print(f"--- wrote result to {fp}")
    return history_list


def train(train_id: str, train_config: TrainConfig):
    print("== running", train_id)
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
                                        deepModel=model, epochs=train_config.epochs, iterations=train_config.iterations,
                                        trainset=ts, ylim_min=train_config.ylim_min, ylim_max=train_config.ylim_max,
                                        yscale=train_config.yscale)
                    history_list = train_histories(training=training)
                    # plot_loss_during_training(training=training, history_list=history_list)
                    print("-" * 40)
                    print(f"- finished training {training.id}")
                    print(f"- with model {training.deepModel.id}")
                    print(f"- on trainset {training.trainset.id}")


if __name__ == '__main__':
    tid = 'nextkarl02a'
    train(tid, train_config=configs[tid])
