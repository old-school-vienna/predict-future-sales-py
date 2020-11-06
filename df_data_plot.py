from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import helpers as hlp
import df_data_cv as dcv


@dataclass
class PlotConfig:
    run_id: str
    ymin_log: float = 0.000001
    ymax_log: float = 0.1
    ymin_linear: float = 0
    ymax_linear: float = 0.001
    desc: str = ''


def plot():
    def plot_id(plot_config: PlotConfig):
        fnam = hlp.dd() / f"cv_results_{plot_config.run_id}.csv"
        if fnam.exists():
            print("-- reading", fnam)
            dat = pd.read_csv(fnam, index_col=0)
            # print(dat.shape)
            # print(dat.keys())
            # print(dat.head())

            fig: Figure = plt.figure()
            fig.set_size_inches(15, 15)
            fig.suptitle(f"cross validation run_id:{plot_config.run_id} \n{plot_config.desc}")
            fig.tight_layout()

            axs = fig.subplots(ncols=2, nrows=2)

            ax: Axes = axs[0][0]
            ax.set_yscale('log')
            ax.set_ylim(plot_config.ymin_log, plot_config.ymax_log)
            ax.violinplot(dat, showmeans=True)
            ax.set_xticks(range(1, len(dat.keys()) + 1))
            ax.set_xticklabels(dat.keys())
            ax.tick_params(axis='x', labelrotation=45.0)

            ax: Axes = axs[0][1]
            ax.set_yscale('log')
            ax.set_ylim(plot_config.ymin_log, plot_config.ymax_log)
            ax.boxplot(dat)
            ax.set_xticks(range(1, len(dat.keys()) + 1))
            ax.set_xticklabels(dat.keys())
            ax.tick_params(axis='x', labelrotation=45.0)

            ax: Axes = axs[1][0]
            ax.set_yscale('linear')
            ax.set_ylim(plot_config.ymin_linear, plot_config.ymax_linear)
            ax.violinplot(dat, showmeans=True)
            ax.set_xticks(range(1, len(dat.keys()) + 1))
            ax.set_xticklabels(dat.keys())
            ax.tick_params(axis='x', labelrotation=45.0)

            ax: Axes = axs[1][1]
            ax.set_yscale('linear')
            ax.set_ylim(plot_config.ymin_linear, plot_config.ymax_linear)
            ax.boxplot(dat)
            ax.set_xticks(range(1, len(dat.keys()) + 1))
            ax.set_xticklabels(dat.keys())
            ax.tick_params(axis='x', labelrotation=45.0)

            fnam = hlp.dd() / f"cv_plot_{plot_config.run_id}.svg"
            fig.savefig(fnam)

            print("-- wrote to", fnam)
        else:
            print("-- not found", fnam)

    plots = [
        PlotConfig('x', desc='data:flat act:relu compare NNs with different complexity'),
        PlotConfig('01', desc='data:flat compare relu(r) and tanh(t) activation'),
        PlotConfig('02', desc='data:flat act:relu'),
        PlotConfig('03', desc='data:flat compare NNs with very small hidden layers'),
        PlotConfig('04', desc='data:flat compare NNs with very small hidden layers'),
        PlotConfig('08', desc='data:flat compare NNs with different depth'),
        PlotConfig('09', desc='data:flat compare NNs with different depth'),
        PlotConfig('struct_01', desc='data:struct compare NNs with different depth 0 - 5',
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('struct_02',
                   desc='data:struct compare NNs with different depth 0 - 10 and size of hidden layers (L, M, S)',
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('struct_02a',
                   desc=dcv.run_configs['struct_02a'].description,
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('struct_02b',
                   desc=dcv.run_configs['struct_02b'].description,
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('s02', desc='data:struct compare NNs with different depth 0 - 3',
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('s03', desc='data:struct compare NNs with different depth 0 - 4',
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('s04', desc='data:struct compare NNs with different depth and small hidden layers',
                   ymin_log=0.00007, ymax_log=0.001, ymax_linear=0.0004),
        PlotConfig('nn01', desc='not normalized',
                   ymin_log=50, ymax_log=500, ymax_linear=500),
        PlotConfig('nn02', desc='not normalized',
                   ymin_log=50, ymax_log=500, ymax_linear=500),
        PlotConfig('nn03', desc='not normalized',
                   ymin_log=50, ymax_log=500, ymax_linear=500),
    ]

    for p in plots:
        plot_id(p)


if __name__ == '__main__':
    plot()
