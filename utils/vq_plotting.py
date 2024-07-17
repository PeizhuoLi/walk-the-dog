import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA


def get_colors(num, colors=mcolors.TABLEAU_COLORS):
    names = list(colors)
    return [colors[names[i % len(names)]] for i in range(num)]


def discrete_color(color_idx):
    palette = get_colors(np.max(color_idx) + 1)
    return [palette[i] for i in color_idx]


def jet_color(x):
    x = np.clip(x, 0, 1)
    jet_map = mpl.cm.get_cmap('jet')
    return jet_map(x)


def plot_embedding(ax, embeddings, values):
    ax.cla()
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings2d = pca.transform(embeddings)
    dot_size = 1

    if np.issubdtype(values.dtype, np.integer):
        n_colors = np.max(values) + 1
        cmap = mcolors.ListedColormap(get_colors(n_colors))
        placeholder = np.zeros((n_colors, 2))
        scatter_legend = ax.scatter(placeholder[:, 0], placeholder[:, 1], c=np.arange(n_colors), cmap=cmap, s=0.)
        colors = discrete_color(values)
        scatter = ax.scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=colors, s=dot_size)
        labels = list(range(n_colors))
        labels = [bin(i)[2:].zfill(2) for i in labels]
        ax.legend(handles=scatter_legend.legend_elements()[0], labels=labels)
    else:
        cax = ax.inset_axes([1.03, 0, 0.01, 1], transform=ax.transAxes)
        scatter = ax.scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=values, cmap='jet', s=dot_size)
        colorbar = plt.colorbar(scatter, ax=ax, cax=cax)
        colorbar.set_label('Relative usage')

    return embeddings2d


def plot_usage_freq(ax, usage):
    ax.cla()
    ax.hist(usage, bins=[0, 0.9, 1.9, 2.9, 3.9])
