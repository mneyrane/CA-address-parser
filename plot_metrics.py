import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

logpath = sys.argv[1]

with open(logpath, 'r') as logfile:
    logs = pd.Series(logfile, dtype='string')

logs = logs.str.rstrip() # remove trailing newlines

#
# extract training accuracy
#
train_mask = logs.str.match('\[TRAIN\]')

train_stats = logs[train_mask].str.extract(
    '[0-9.]+.*?[0-9.]+.*?([0-9.]+).*?([0-9.]+)')

train_stats = train_stats.astype(float)

train_stats = train_stats.rename({
    0 : 'per-char', 
    1 : 'parser',
    }, axis='columns')

train_stats = train_stats.reset_index(drop=True)

train_stats.insert(0, 'batch', (train_stats.index // 100) * 100)

train_results = pd.melt(
    train_stats, 
    id_vars=['batch'], 
    var_name='metric', 
    value_name='accuracy')

#
# extract test accuracy
#
test_mask = logs.str.match('\[TEST\]')

test_stats = logs[test_mask].str.extract(
    '[0-9.]+.*?([0-9.]+).*?([0-9.]+).*?([0-9.]+).*?([0-9.]+)')

test_stats = test_stats.astype(float)

test_stats = test_stats.rename({
    0 : 'per-char (generated)',
    1 : 'parser (generated)',
    2 : 'per-char (real)',
    3 : 'parser (real)',
    }, axis='columns')

test_stats = test_stats.reset_index(drop=True)
test_stats = test_stats.reset_index(names='epoch')

test_results = pd.melt(
    test_stats,
    id_vars=['epoch'],
    var_name='metric',
    value_name='accuracy')

#
# plot results
#

sns.set_theme(context='paper', style='whitegrid', font='Arimo')
facecolor = '#f8f5f0'

fig, axs = plt.subplots(1, 2, figsize=(8,3), facecolor=facecolor)

# training accuracy
sns.lineplot(data=train_results, x='batch', y='accuracy', hue='metric', errorbar=('pi', 100), ax=axs[0])
axs[0].set_xlabel('Batch number')
axs[0].set_xticks([0,15000,30000,45000,60000,75000])
axs[0].set_ylabel('Accuracy (%)')
axs[0].set_title('training')

# test accuracy
sns.lineplot(data=test_results, x='epoch', y='accuracy', hue='metric', ax=axs[1])
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].set_title('test')

fig.tight_layout()
fig.savefig(fig_dir / 'metric_plots.svg', bbox_inches='tight')
