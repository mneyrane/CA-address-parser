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
    0 : 'per-char (PG)',
    1 : 'parser (PG)',
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

# training accuracy
plt.figure(facecolor=facecolor)
ax = sns.lineplot(data=train_results, x='batch', y='accuracy', hue='metric', errorbar=('pi', 100))
ax.set_xlabel('Batch number')
ax.set_ylabel('Accuracy (%)')
ax.set_title('CCAPNet training metrics')
plt.savefig(fig_dir / 'train_plot.svg', bbox_inches='tight')

# test accuracy
plt.figure(facecolor=facecolor)
ax = sns.lineplot(data=test_results, x='epoch', y='accuracy', hue='metric')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('CCAPNet test metrics')
plt.savefig(fig_dir / 'test_plot.svg', bbox_inches='tight')
