import scienceplots
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'grid'])

num_frac_bits = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
acc_vs_epoch = [0.788761707, 0.807492196, 0.807492196, 0.803329865, 0.803329865, 0.803329865, 0.804370447, 0.803329865, 0.804370447, 0.803329865, 0.803329865, 0.804370447, 0.740894901, 0.563995838]

pparam = dict(xlabel='Number of fractional bits', ylabel='Validation accuracy')

with plt.style.context(['science']):
    plt.rcParams['grid.linewidth'] = 0.5
    fig, ax = plt.subplots()
    ax.plot(num_frac_bits, acc_vs_epoch, marker='o', markersize=2, label='Validation accuracy')
    ax.axhline(y=0.817898023, color='black', linestyle='--', alpha=0.5, label='TensorFlow reference')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0.5, 1) # Set the limits of y-axis from 0 to 1
    ax.legend()  # Add a legend
    fig.savefig('fixed_point_acc.eps', format='eps')
    fig.savefig('fixed_point_acc.png', format='png', dpi=600)
    plt.close()
