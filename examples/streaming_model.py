from pancakes.models.densitysplit import DensitySplitCCF
from pancakes.utilities.utilities import get_multipoles
from pancakes.perturbation_theory.empirical import radial_velocity
import numpy as np
import matplotlib.pyplot as plt

params_file = 'pancakes_params.yaml'

ds = DensitySplitCCF(params_file)

s = np.linspace(0, 150, 30)
mu = np.linspace(-1, 1, 100)

theory_xi = ds.theory_xi_smu(
    fs8=ds.fs8,
    bs8=ds.bs8,
    sigma_v=360,
    q_perp=1.0,
    q_para=1.0,
    s=s,
    mu=mu,
    denbin='DS1',
    nu=0.29
)

theory_xi0, theory_xi2 = get_multipoles([0, 2], s, mu,
                                       theory_xi)

# fig, ax = plt.subplots(figsize=(4.5, 4.5))

# ax.plot(s, s**2*theory_xi0)

# ax.set_xlim(0, 150)
# plt.show()

s_perp = np.linspace(0, 150, 100)
s_para = np.linspace(0, 150, 100)

theory_xi = ds.theory_xi_sigmapi(
    fs8=ds.fs8,
    bs8=ds.bs8,
    sigma_v=360,
    q_perp=1.0,
    q_para=1.0,
    s_perp=s_perp,
    s_para=s_para,
    nu = 0.29,
    denbin='DS1',
    squared=True
)

fig, ax = plt.subplots(figsize=(5, 5))

# ax.imshow(
#     theory_xi, aspect='equal',
#     extent=(s_perp.min(),s_perp.max(),
#     s_para.min(), s_para.max())
# )

ax.contour(s_perp, s_para, theory_xi)

plt.show()
