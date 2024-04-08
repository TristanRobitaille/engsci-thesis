import scienceplots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use(['science', 'grid'])

data = [
    {
        "name": "mem_aspect_ratio_4",
        "mem_capacity_word": [128, 256, 512, 1024, 2048],
        "mem_area_um2": [7765.012, 9535.5722, 14249.134, 24558.773, 45178.052],
        "overhead": 7570
    },
    {
        "name": "mem_aspect_ratio_8",
        "mem_capacity_word": [256, 512, 1024, 2048, 4096],
        "mem_area_um2": [9738.926, 12673.166, 20420.48, 36413.001, 68398.051],
        "overhead": 7600
    },
    {
        "name": "mem_aspect_ratio_16",
        "mem_capacity_word": [512, 1024, 2048, 4096, 8192],
        "mem_area_um2": [16876.238, 21960.878, 35385.92, 63098.793, 118524.547],
        "overhead": 13060
    },
    {
        "name": "mem_aspect_ratio_32",
        "mem_capacity_word": [1024, 2048, 4096, 8192, 16384],
        "mem_area_um2": [31150.861, 40536.301, 65316.79, 116470.377, 218777.539],
        "overhead": 26320
    }
]

# Area plot
with plt.style.context(['science']):
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust grid line width
    fig, ax = plt.subplots()
    pparam = dict(xlabel='Memory size ($\\times$16 bits/word)', ylabel='Memory area ($\mu m^{2}$)')
    
    for line in data:
        if line["name"] != "mem_aspect_ratio_4": ax.plot(line["mem_capacity_word"], line["mem_area_um2"], linestyle='--', alpha=0.5)
        else: ax.plot(line["mem_capacity_word"], line["mem_area_um2"], linestyle='-')

    ax.axvline(x=848, color='black', alpha=0.75, linewidth=0.5) # Memory size for intermediate results
    ax.axvline(x=528, color='black', alpha=0.75, linewidth=0.5) # Memory size for model parameters

    legend = ax.legend(['Aspect ratio = 4', 'Aspect ratio = 8', 'Aspect ratio = 16', 'Aspect ratio = 32'], loc='lower right', fontsize='smaller')
    ax.add_artist(legend)

    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)

    ax.set(**pparam)
    fig.savefig('mem_area.eps', format='eps')
    fig.savefig('mem_area.png', format='png', dpi=600)
    plt.close()

# Overhead plot
with plt.style.context(['science']):
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust grid line width
    fig, ax = plt.subplots()
    pparam = dict(xlabel='Memory size ($\\times$16 bits/word)', ylabel='Overhead ratio')

    for line in data:
        if line["name"] != "mem_aspect_ratio_4": 
            overhead = line['overhead']
            overhead_list = [overhead] * len(line['mem_area_um2'])
            ax.plot(line["mem_capacity_word"], np.array(overhead_list) / np.array(line['mem_area_um2']), linestyle='--', alpha=0.5)
        else: 
            overhead = line['overhead']
            overhead_list = [overhead] * len(line['mem_area_um2'])
            ax.plot(line["mem_capacity_word"], np.array(overhead_list) / np.array(line['mem_area_um2']), linestyle='-')

    ax.axvline(x=848, color='black', alpha=0.75, linewidth=0.5) # Memory size for intermediate results
    ax.axvline(x=528, color='black', alpha=0.75, linewidth=0.5) # Memory size for model parameters

    legend = ax.legend(['Aspect ratio = 4', 'Aspect ratio = 8', 'Aspect ratio = 16', 'Aspect ratio = 32'], loc='upper right', fontsize='smaller')
    ax.add_artist(legend)

    ax.set_ylim(0, 1)
    ax.set_xscale('log', base=2)

    ax.set(**pparam)
    fig.savefig('mem_overhead.eps', format='eps')
    fig.savefig('mem_overhead.png', format='png', dpi=600)
    plt.close()
