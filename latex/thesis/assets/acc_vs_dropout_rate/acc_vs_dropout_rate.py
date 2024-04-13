import scienceplots
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'grid'])

dropout_rate = np.linspace(0.05, 0.95, 19)
acc = [0.7941, 0.7988, 0.782, 0.8046, 0.8009, 0.8046, 0.7894, 0.7075, 0.7237, 0.6891, 0.6833, 0.438, 0.1891, 0.2568, 0.177, 0.2258, 0.4758, 0.5914, 0.5914]

pparam = dict(xlabel='Dropout rate', ylabel='Validation accuracy')

with plt.style.context(['science']):
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust grid line width
    fig, ax = plt.subplots()
    ax.plot(dropout_rate, acc, marker='o', markersize=2, color='b', label='Validation accuracy')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0, 1)  # Set the limits of y-axis from 0 to 1
    fig.savefig('acc_vs_dropout_rate.eps', format='eps')
    fig.savefig('acc_vs_dropout_rate.png', format='png', dpi=600)
    plt.close()
