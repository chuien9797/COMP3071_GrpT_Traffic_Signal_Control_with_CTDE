import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi  = dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):

        if len(data) == 0:
            print(f"[Viz]  '{filename}': no data, skipping figure.")  # optional log
            return

        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, f"plot_{filename}.png"), dpi=self._dpi)
        plt.close(fig)

        with open(os.path.join(self._path, f"plot_{filename}_data.txt"), "w") as fh:
            for v in data:
                fh.write(f"{v}\n")