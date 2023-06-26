import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

from train import label_chars
from address.proc_gen import TokenCategory

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

cmpath = sys.argv[1]

#
# load confusion matrices
#
with np.load(cmpath, 'r') as data:
    cm = dict(data)

cm_pg = pd.DataFrame(cm['pg'])
cm_real = pd.DataFrame(cm['real'])

mapping = {cat.value : cat.name for cat in TokenCategory}

cm_pg.rename(index=mapping, columns=mapping, inplace=True)
cm_real.rename(index=mapping, columns=mapping, inplace=True)

#
# plot results
#
sns.set_theme(context='paper', font='Arimo')
facecolor = '#f8f5f0'
linewidth = .5
norm = clr.LogNorm()

cmap = sns.color_palette('crest', as_cmap=True)
cmap.set_bad('lightgrey')

fig, axs = plt.subplots(1, 2, figsize=(7.25,3), facecolor=facecolor)

# confusion matrix for procedurally generated data
sns.heatmap(cm_pg, linewidth=linewidth, cmap=cmap, norm=norm, square=True, ax=axs[0])
axs[0].set_title('generated')

# confusion matrix for real world data
sns.heatmap(cm_real, linewidth=linewidth, cmap=cmap, norm=norm, square=True, ax=axs[1])
axs[1].set_title('real')

fig.tight_layout()
fig.savefig(fig_dir / 'cm.svg', bbox_inches='tight')
