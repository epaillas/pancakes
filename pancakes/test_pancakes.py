from models.densitysplit_ccf import DensitySplitCCF
from utilities.utilities import get_multipoles
from perturbation_theory.empirical import radial_velocity
import numpy as np
import matplotlib.pyplot as plt

xi_r_filename_i = 'example_data/Galaxies_HOD_z0.57_Real_den1.tpcf'
xi_r_filename_j = 'example_data/Galaxies_HOD_z0.57_Real_den5.tpcf'
xi_r_filename = f'{xi_r_filename_i}'

sv_rmu_filename_i = 'example_data/Galaxies_HOD_z0.57_Real_den1.std_los_velocity_vs_rmu'
sv_rmu_filename_j = 'example_data/Galaxies_HOD_z0.57_Real_den5.std_los_velocity_vs_rmu'
sv_rmu_filename = f'{sv_rmu_filename_i}'

smin = '0'
smax = '150'

ds = DensitySplitCCF(
    xi_r_filename=xi_r_filename,
    sv_rmu_filename=sv_rmu_filename,
    smin=smin,
    smax=smax)

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
    denbin='den0',
    nu=0.29)

theory_xi0, theory_xi2 = get_multipoles([0, 2], s, mu,
                                       theory_xi)

fig, ax = plt.subplots(figsize=(4.5, 4.5))

ax.plot(s, s**2*theory_xi0)

ax.set_xlim(0, 150)
plt.show()


xi_r = np.genfromtxt(xi_r_filename_i)


