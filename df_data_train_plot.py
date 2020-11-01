from pathlib import Path
import helpers as hlp
import re
import json
import matplotlib.pylab as plt


def plot():
    pattern = f"train.*json"
    base_dir = hlp.dd() / 'nextkarl02b'

    def plot_json(json_file: Path):
        def plot_training(train_result: dict):
            stem = json_file.stem
            train_id = train_result['id']
            data = train_result['data']
            model = train_result['model']
            title = f"training {train_id} with model {model} on {data}"
            histories = train_result['histories']

            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            ax.set_title(title)
            ax.set_ylim(100, 200)
            ax.set_yscale('linear')
            for hist in histories:
                ax.plot(hist)

            parent = json_file.parent
            plot_path = parent / f"{stem}.svg"
            fig.savefig(plot_path)
            print("-- saved figure to", plot_path)

        print("--- plotting", json_file)
        with open(json_file) as f:
            plot_training(json.load(f))

    files = [f for f in base_dir.iterdir() if re.match(pattern, f.name)]
    files = sorted(files, key=lambda f: f.name)
    [plot_json(f) for f in files]
