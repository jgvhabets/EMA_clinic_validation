
import numpy as np


def get_sub_colors(data):
    
    import matplotlib as mpl

    allsubs = []
    for mom in list(data.keys()): allsubs.extend(list(data[mom].keys()))
    allsubs = np.unique(allsubs)

    colors1 = list(mpl.colormaps['Set3'].colors)
    colors2 = list(mpl.colormaps['Set2'].colors)
    colors = colors1 + colors2

    subcolors = {s: colors[i] for i, s in enumerate(allsubs)}

    return subcolors