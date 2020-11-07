from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import df_data_cv as dcv
import helpers as hlp


@dataclass
class PlotScale:
    scale: str
    ymin: float
    ymax: float


@dataclass
class PlotConfig:
    run_id: str
    plot_scales: List[PlotScale]
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

            subcnt = len(plot_config.plot_scales)
            fig: Figure = plt.figure()
            fig.set_size_inches(16, 8 * subcnt)
            fig.suptitle(f"cross validation run_id:{plot_config.run_id} \n{plot_config.desc}")
            fig.tight_layout()

            axs = fig.subplots(ncols=2, nrows=subcnt)
            for i in range(0, subcnt):
                ps: PlotScale = plot_config.plot_scales[i]

                if subcnt == 1:
                    ax: Axes = axs[0]
                else:
                    ax: Axes = axs[i][0]
                ax.set_yscale(ps.scale)
                ax.set_ylim(ps.ymin, ps.ymax)
                ax.violinplot(dat, showmeans=True)
                ax.set_xticks(range(1, len(dat.keys()) + 1))
                ax.set_xticklabels(dat.keys())
                ax.tick_params(axis='x', labelrotation=45.0)

                if subcnt == 1:
                    ax: Axes = axs[1]
                else:
                    ax: Axes = axs[i][1]
                ax.set_yscale(ps.scale)
                ax.set_ylim(ps.ymin, ps.ymax)
                ax.boxplot(dat)
                ax.set_xticks(range(1, len(dat.keys()) + 1))
                ax.set_xticklabels(dat.keys())
                ax.tick_params(axis='x', labelrotation=45.0)

            fnam = hlp.dd() / f"cv_plot_{plot_config.run_id}.svg"
            fig.savefig(fnam)

            print("-- wrote to", fnam)
        else:
            print("-- not found", fnam)

    default_plot_scales = [
        PlotScale(
            scale='linear',
            ymin=0,
            ymax=0.001,
        ),
        PlotScale(
            scale='log',
            ymin=0.000001,
            ymax=0.1,
        ),
    ]

    rel1_plot_scales = [
        PlotScale(
            scale='linear',
            ymin=0,
            ymax=0.00041,
        ),
        PlotScale(
            scale='log',
            ymin=0.00007,
            ymax=0.01,
        ),
    ]

    norm_plot_scales = [
        PlotScale(
            scale='linear',
            ymin=100,
            ymax=800,
        ),
    ]

    plots = [
        PlotConfig('x', desc='data:flat act:relu compare NNs with different complexity',
                   plot_scales=default_plot_scales),
        PlotConfig('01', desc='data:flat compare relu(r) and tanh(t) activation', plot_scales=default_plot_scales),
        PlotConfig('02', desc='data:flat act:relu', plot_scales=default_plot_scales),
        PlotConfig('03', desc='data:flat compare NNs with very small hidden layers', plot_scales=default_plot_scales),
        PlotConfig('04', desc='data:flat compare NNs with very small hidden layers', plot_scales=default_plot_scales),
        PlotConfig('08', desc='data:flat compare NNs with different depth', plot_scales=default_plot_scales),
        PlotConfig('09', desc='data:flat compare NNs with different depth', plot_scales=default_plot_scales),
        PlotConfig('struct_01', desc='data:struct compare NNs with different depth 0 - 5',
                   plot_scales=rel1_plot_scales),
        PlotConfig('struct_02',
                   desc='data:struct compare NNs with different depth 0 - 10 and size of hidden layers (L, M, S)',
                   plot_scales=rel1_plot_scales),
        PlotConfig('struct_02a',
                   desc=dcv.run_configs['struct_02a'].description,
                   plot_scales=rel1_plot_scales),
        PlotConfig('struct_02b',
                   desc=dcv.run_configs['struct_02b'].description,
                   plot_scales=rel1_plot_scales),
        PlotConfig('s02', desc='data:struct compare NNs with different depth 0 - 3',
                   plot_scales=rel1_plot_scales),
        PlotConfig('s03', desc='data:struct compare NNs with different depth 0 - 4',
                   plot_scales=rel1_plot_scales),
        PlotConfig('s04', desc='data:struct compare NNs with different depth and small hidden layers',
                   plot_scales=rel1_plot_scales),
        PlotConfig('nn01', desc='not normalized 1', plot_scales=norm_plot_scales),
        PlotConfig('nn02', desc='not normalized 2', plot_scales=norm_plot_scales),
        PlotConfig('nn03', desc='not normalized 3', plot_scales=norm_plot_scales),
        PlotConfig('nh01', desc='not normalized/hot 1', plot_scales=norm_plot_scales),
    ]

    for p in plots:
        plot_id(p)


if __name__ == '__main__':
    plot()
