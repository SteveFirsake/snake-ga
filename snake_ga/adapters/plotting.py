import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from snake_ga.application.ports import ScorePlotterPort


class SeabornScorePlotter(ScorePlotterPort):
    def plot(self, games: list[int], scores: list[float], train: bool) -> None:
        sns.set(color_codes=True, font_scale=1.5)
        sns.set_style("white")
        plt.figure(figsize=(13, 8))
        fit_reg = not train
        ax = sns.regplot(
            np.array([games])[0],
            np.array([scores])[0],
            x_jitter=0.1,
            scatter_kws={"color": "#36688D"},
            label="Data",
            fit_reg=fit_reg,
            line_kws={"color": "#F49F05"},
        )
        y_mean = [np.mean(scores)] * len(games)
        ax.plot(games, y_mean, label="Mean", linestyle="--")
        ax.legend(loc="upper right")
        ax.set(xlabel="# games", ylabel="score")
        plt.show()
