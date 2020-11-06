from dataclasses import dataclass
from typing import List, Callable

import pandas as pd
import sklearn.model_selection as ms

import df_data_flat as fdat
import df_data_struct as sdat
import df_data_next as ndat
import helpers as hlp

import argparse as ap


@dataclass
class NN:
    id: str
    activation: str
    layers: list


@dataclass
class RunConfig:
    count_per_config: int
    epochs: int
    batch_size: int
    nns: List[NN]
    trainset: Callable
    description: str = ''


run_configs = {
    'f09': RunConfig(
        count_per_config=30,
        epochs=20,
        batch_size=10,
        nns=[
            NN('h_4', 'relu', [0.2, 0.1, 0.05, 0.025, 0.012]),
            NN('h_5', 'relu', [0.2, 0.1, 0.05, 0.025, 0.012, 0.01]),
            NN('h_6', 'relu', [0.2, 0.1, 0.05, 0.025, 0.012, 0.01, 0.007]),
            NN('h_7', 'relu', [0.2, 0.1, 0.05, 0.025, 0.012, 0.01, 0.007, 0.005]),
            NN('h_8', 'relu', [0.2, 0.1, 0.05, 0.025, 0.012, 0.01, 0.007, 0.005, 0.001]),
        ],
        trainset=fdat.read_train_data
    ),
    'struct_02': RunConfig(
        description='data:struct compare NNs with different depth 0 - 10 and size of hidden layers (L, M, S)',
        count_per_config=20,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_00', 'relu', []),
            NN('h_01_L', 'relu', [0.7]),
            NN('h_01_M', 'relu', [0.5]),
            NN('h_01_S', 'relu', [0.3]),
            NN('h_01', 'relu', [0.7]),
            NN('h_02', 'relu', [0.7, 0.5]),
            NN('h_03', 'relu', [0.7, 0.5, 0.2]),
            NN('h_04', 'relu', [0.7, 0.5, 0.2, 0.1]),
            NN('h_05', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1]),
            NN('h_06', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1]),
            NN('h_07', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1]),
            NN('h_08', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
            NN('h_09', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            NN('h_10', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    'struct_02a': RunConfig(
        description='data:struct compare NNs with different size of hidden layers (L, M, S)',
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_00', 'relu', []),
            NN('h_01_L', 'relu', [0.7]),
            NN('h_01_M', 'relu', [0.5]),
            NN('h_01_S', 'relu', [0.3]),
            NN('h_02_L', 'relu', [0.7, 0.5]),
            NN('h_02_M', 'relu', [0.5, 0.3]),
            NN('h_02_S', 'relu', [0.3, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    'struct_02b': RunConfig(
        description='data:struct compare NNs with different depth 0 - 10',
        count_per_config=40,
        epochs=6,
        batch_size=10,
        nns=[
            NN('h_00', 'relu', []),
            NN('h_01', 'relu', [0.7]),
            NN('h_02', 'relu', [0.7, 0.5]),
            NN('h_03', 'relu', [0.7, 0.5, 0.2]),
            NN('h_04', 'relu', [0.7, 0.5, 0.2, 0.1]),
            NN('h_05', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1]),
            NN('h_06', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1]),
            NN('h_07', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1]),
            NN('h_08', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
            NN('h_09', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            NN('h_10', 'relu', [0.7, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    's02': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0', 'relu', []),
            NN('h_1', 'relu', [0.2]),
            NN('h_2', 'relu', [0.2, 0.1]),
            NN('h_3', 'relu', [0.2, 0.1, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    's03': RunConfig(
        count_per_config=20,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0', 'tanh', []),
            NN('h_1', 'tanh', [0.2]),
            NN('h_2', 'tanh', [0.2, 0.1]),
            NN('h_3', 'tanh', [0.2, 0.1, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    's04': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0L', 'relu', []),
            NN('h_1L', 'relu', [0.7]),
            NN('h_2L', 'relu', [0.7, 0.5]),
            NN('h_3L', 'relu', [0.7, 0.5, 0.2]),
            NN('h_4L', 'relu', [0.7, 0.5, 0.2, 0.1]),
        ],
        trainset=sdat.read_train_data
    ),
    'nn01': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0L', 'relu', []),
            NN('h_1L', 'relu', [0.5]),
            NN('h_2L', 'relu', [1.0]),
            NN('h_3L', 'relu', [0.5, 0.2]),
            NN('h_4L', 'relu', [1.0, 1.0]),
        ],
        trainset=ndat.read_train_data_karl_not_norm
    ),
    'nn02': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0', 'relu', []),
            NN('h_1a', 'relu', [0.1]),
            NN('h_1b', 'relu', [0.5]),
            NN('h_1c', 'relu', [1.0]),
            NN('h_3a', 'relu', [0.1, 0.1]),
            NN('h_3b', 'relu', [0.3, 0.1]),
            NN('h_3c', 'relu', [0.5, 0.1]),
            NN('h_4d', 'relu', [1.0, 1.5]),
        ],
        trainset=ndat.read_train_data_karl_not_norm
    ),
    'nn03': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('h_0', 'relu', []),

            NN('h_1a', 'relu', [0.1]),
            NN('h_1b', 'relu', [0.5]),
            NN('h_1c', 'relu', [1.0]),

            NN('h_3a', 'relu', [0.1, 0.1]),
            NN('h_3b', 'relu', [0.5, 0.3]),
            NN('h_3c', 'relu', [1.0, 0.5]),
            NN('h_3d', 'relu', [1.0, 1.0]),

            NN('h_4a', 'relu', [0.1, 0.1, 0.1]),
            NN('h_4b', 'relu', [0.5, 0.3, 0.1]),
            NN('h_4c', 'relu', [1.0, 0.5, 0.3]),
            NN('h_4d', 'relu', [1.0, 1.0, 1.0]),
        ],
        trainset=ndat.read_train_data_karl_not_hot
    ),
    'nh01': RunConfig(
        count_per_config=40,
        epochs=5,
        batch_size=10,
        nns=[
            NN('nh_0', 'relu', []),

            NN('nh_1a', 'relu', [0.1]),
            NN('nh_1b', 'relu', [0.5]),
            NN('nh_1c', 'relu', [1.0]),

            NN('h_2a', 'relu', [0.1, 0.1]),
            NN('h_2b', 'relu', [0.5, 0.3]),
            NN('h_2c', 'relu', [1.0, 1.0]),

            NN('h_3a', 'relu', [0.1, 0.1, 0.1]),
            NN('h_3b', 'relu', [0.5, 0.3, 0.1]),
            NN('h_3c', 'relu', [1.0, 1.0, 1.0]),
        ],
        trainset=ndat.read_train_data_karl_not_hot
    ),
    'small': RunConfig(
        count_per_config=3,
        epochs=5,
        batch_size=10,
        nns=[
            NN('nh_0', 'relu', []),
        ],
        trainset=ndat.read_train_data_karl_not_hot
    )
}


def cv(run_id: str, run_config: RunConfig):
    print(f"==== running {run_id}")

    trainset = run_config.trainset()
    print("-- x", trainset.x.shape)
    print("-- y", trainset.y.shape)

    def cv_one(nn: NN) -> float:
        x_train, x_test, y_train, y_test = ms.train_test_split(trainset.x, trainset.y)
        print("-- x train", x_train.shape)
        print("-- y train", y_train.shape)
        print("-- x test", x_test.shape)
        print("-- y test", y_test.shape)

        layer_configs = [hlp.LayerConfig(size) for size in nn.layers]
        model_config = hlp.ModelConfig(activation=nn.activation, optimizer='adam',
                                       loss='mean_squared_error', layers=layer_configs)
        model = hlp.create_model(model_config, x_train.shape[1])
        history = model.fit(x_train, y_train, epochs=run_config.epochs, batch_size=run_config.batch_size)
        for loss in history.history['loss']:
            print(f'-- {loss:.6f}')

        err = model.evaluate(x_test, y_test)
        return err

    def multi(nn: NN) -> (str, list):
        errs1 = [cv_one(nn) for _ in range(run_config.count_per_config)]
        return nn.id, errs1

    results = [multi(nn) for nn in run_config.nns]
    for id1, errs in results:
        print(f'== {id1} ==')
        [print(f"{err:.6f}") for err in errs]

    df_results = pd.DataFrame(dict(results))

    fnam = hlp.dd() / f"cv_results_{run_id}.csv"
    df_results.to_csv(fnam)
    print("-- wrote to", fnam)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Run a crossvalidation')
    parser.add_argument(dest='run_id', type=str, help='a run id defining a configuration')
    ri = parser.parse_args().run_id
    if ri in run_configs:
        cv(ri, run_configs[ri])
    else:
        known = ','.join([i for i in run_configs.keys()])
        print(f"Unknown run_id '{ri}'. ({known})")
