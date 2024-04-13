import scienceplots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use(['science', 'grid'])

hyperparam = {
    "emb_depth":[16, 32, 64, 128],
    "mlp_dim":[8, 16, 32, 64],
    "enc_layer":[1, 2, 3, 4],
    "att_head":[4, 8, 16, 32],
    "sampling_freq":[64, 128, 256],
    "patch_len": [16, 32, 64, 128, 256, 512]
}

acc = {
    "emb_depth":[0.799, 0.8217, 0.828, 0.828],
    "mlp_dim":[0.8173, 0.8252, 0.8296, 0.824],
    "enc_layer":[0.824, 0.818, 0.818, 0.821],
    "att_head":[0.823, 0.822, 0.827, 0.824],
    "sampling_freq":[0.808, 0.823, 0.818],
    "patch_len": [0.7243, 0.7668, 0.7915, 0.7941, 0.7962, 0.7568]
}

xlabels = {
    "emb_depth":"Embedding depth",
    "mlp_dim":"MLP dimension",
    "enc_layer":"Ecoder layers",
    "att_head":"Attention heads",
    "sampling_freq":"Sampling frequency (Hz)",
    "patch_len": "{Patch length}"
}

pparam = dict(xlabel='Dropout rate', ylabel='Validation accuracy')

for key in acc.keys():
    with plt.style.context(['science']):
        plt.rcParams['grid.linewidth'] = 0.5
        fig, ax = plt.subplots()
        ax.plot(hyperparam[key], acc[key], marker='o', markersize=2, color='b', label='Validation accuracy')
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabels[key])
        ax.set_ylabel('Validation accuracy')
        ax.set_ylim(0.5, 1)
        ax.set_xticks(hyperparam[key])
        if key != "enc_layer":
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.grid(True)
        fig.savefig(f'{key}.eps', format='eps')
        fig.savefig(f'{key}.png', format='png', dpi=600)
        plt.close()