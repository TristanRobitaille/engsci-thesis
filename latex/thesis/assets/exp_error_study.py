import scienceplots
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'grid'])

MAX_ORDER = 10
order_taylor = np.linspace(0, MAX_ORDER-1, MAX_ORDER)
rel_error_float_vs_float = [0.4998527977758119, 0.1533243985797279, 0.03327780938020009, 0.005552936783181807, 0.0007506227463167017, 8.51633822610256e-05, 8.319149492860161e-06, 7.131861598598643e-07, 5.4459517344677454e-08, 3.74833828046497e-09]
rel_error_fix_vs_float =   [0.4998527977758119, 0.1540885369648442, 0.03593633249821528, 0.011065564588370644, 0.00992885233140691, 0.00992885233140691, 0.00992885233140691, 0.00992885233140691, 0.00992885233140691, 0.00992885233140691]
pparam = dict(xlabel='Order of Taylor series expansion', ylabel='Relative error')

with plt.style.context(['science']):
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust grid line width
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, MAX_ORDER-1, MAX_ORDER), rel_error_float_vs_float)
    ax.plot(np.linspace(0, MAX_ORDER-1, MAX_ORDER), rel_error_fix_vs_float)
    legend = ax.legend(['Float approx.', 'Fixed-point approx.'])
    ax.add_artist(legend)
    ax.legend(bbox_to_anchor=(0., 1.2)) # Move legend up by 10%
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.autoscale(tight=True)
    ax.set(**pparam)
    fig.savefig('exp_approx_error.eps', format='eps')
    fig.savefig('exp_approx_error.png', format='png', dpi=300)
    plt.close()
    