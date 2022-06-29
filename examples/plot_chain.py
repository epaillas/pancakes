import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from getdist import plots, MCSamples
import getdist
import matplotlib
matplotlib.use('TkAgg')
# plt.style.use('enrique')
plt.rcParams['text.usetex'] = True

input_fn = 'data/zero_quadrupole_ap.h5'
names = ['beta', 'epsilon']
labels = [r'\beta', r'\epsilon']

sample_list = []
reader = emcee.backends.HDFBackend(input_fn)
chain = reader.get_chain(flat=True)

for i, paraname in enumerate(names):
    ds_median = np.quantile(chain[:,i], 0.5, axis=0)
    ds_slo = np.quantile(chain[:,i], 0.16, axis=0)
    ds_shi = np.quantile(chain[:,i], 0.84, axis=0)
    ds_err = (ds_shi - ds_median)
    print('DS {} median, err, precision: {} {} {}'.format(
        paraname, ds_median, ds_err, ds_err/ds_median))

samples = MCSamples(samples=chain, labels=labels, names=names)
# p = samples.getParams()
# samples.addDerived(p.f / p.b, name='beta',
#     label=r'\beta')

sample_list.append(samples)

# Triangle plot
g = plots.get_subplot_plotter()
g.settings.axes_fontsize = 16
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 20
g.settings.axis_marker_lw = 1.0
g.settings.title_limit_fontsize=16
g.settings.title_limit_labels = False
g.settings.tight_layout = True
g.settings.axis_marker_color = 'dimgrey'
# param_limits = {'q_para':(0.995, 1.025)}

g.triangle_plot(roots=sample_list,
    params=['beta', 'epsilon'],
    filled=True,
    # legend_labels=chain_labels,
    legend_loc='upper right',
    # line_args=line_args,
    # contour_colors=colors,
    title_limit=0,
    markers={'beta': 0.36, 'epsilon': 1.0})


plt.savefig('test.pdf')
