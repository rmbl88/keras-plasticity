import enum
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import constants
import numpy as np

def update(num):

    lines = []
    for i, line in enumerate(ax.lines[:-1]):

        line.set_data(epochs[:num], losses[i][:num])
        lines.append(line)

    lines.append(ax.lines[-1].set_xdata([num,num]))

    # line_1.set_data(x_1[:num], y_1[:num])
    # line_2.set_data(x_2[:num], y_2[:num])
    #line.axes.axis([0, 10, 0, 1])
    return lines

plt.rcParams.update(constants.PARAMS)

anim_writer=animation.PillowWriter(fps=12, codec='libx264', bitrate=2)

DIR = 'outputs/9-elem/loss/'
ARCH = '[6-8x1-3]'

file_list = []
df_list = []

for r, d, f in os.walk(DIR):
    for file in sorted(f):
        if ARCH in file and '.csv' in file:
            file_list.append(DIR + file)

df_list = [pd.read_csv(file, sep=',', index_col=0) for file in file_list]

fig = plt.figure(figsize=(10,6), constrained_layout = True)
ax = fig.add_subplot(111)

ax.set_xlabel(r'epoch', fontsize=16)
ax.set_ylabel(r'MSE Loss [J\textsuperscript{2}]', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

losses = dict.fromkeys(np.arange(len(file_list)))

for i, df in enumerate(df_list):

    label = file_list[i].split('/')[-1][:-4]
    ax.plot(df['epoch'], df['loss'], label=label, lw=1.5)
    losses[i] = df['loss']
    if i == 0:
        epochs = df['epoch']

ax.axvline(0, ls='-', color='lightgray', lw=0.75, zorder=10)

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels,handles), key=lambda x: int(x[0].split('-')[-2])))

ax.legend(handles, labels)

import matplotlib.ticker as ticker
#ax.xaxis.set_major_locator(ticker.MultipleLocator(5))      
plt.legend(loc='upper right', prop={'size': 16})      
ani = animation.FuncAnimation(fig, update, len(epochs), interval=12, blit=False, repeat_delay=3000, cache_frame_data=False)
ani.save(DIR + ARCH + '_loss_anim.gif', writer=anim_writer, dpi=300)

print('hey')