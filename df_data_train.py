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


configs = {
    'next02': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_s, ndat.read_train_data_m, ndat.read_train_data_l],

    ),
    'nextkarl01': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_karl],
    ),
    'nextkarl02': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[10],
        layers_list=[[], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_karl_not_norm],
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
    ),
    'nextkarl02b': TrainConfig(
        epochs=30,
        iterations=4,
        batch_sizes=[5, 10],
        layers_list=[
            [],
            [1.0],
            [1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        activations=["relu", "tanh"],
        trainsets=[ndat.read_train_data_karl_not_norm],
    ),
    'tryout': TrainConfig(
        epochs=5,
        iterations=2,
        batch_sizes=[5],
        layers_list=[[1.0]],
        activations=["relu"],
        trainsets=[ndat.read_train_data_s],
    )
}


def fnam(training: Training) -> str:
    return f"train_{training.id}_{training.trainset.id}_{training.deepModel.id}"


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
                                        trainset=ts)
                    history_list = train_histories(training=training)
                    print("-" * 40)
                    print(f"- finished training {training.id}")
                    print(f"- with model {training.deepModel.id}")
                    print(f"- on trainset {training.trainset.id}")


if __name__ == '__main__':
    tid = 'nextkarl02b'
    train(tid, train_config=configs[tid])
