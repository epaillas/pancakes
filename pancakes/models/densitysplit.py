import numpy as np
import yaml
import os
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import norm
from ..perturbation_theory.empirical import radial_velocity
from ..utilities.utilities import multipole, read_2darray
from ..utilities.cosmology import Cosmology


class DensitySplitCCF:
    """
    Class to perform RSD & AP fits to the Density Split cross-correlation
    functons as in https://arxiv.org/abs/2101.09854
    """
    def __init__(
        self,
        param_file: str
    ):
        """
        Initialize class.
        Args:
            param_file: YAML file containing configuration parameters.
        """
        with open(param_file) as file:
            self.params = yaml.full_load(file)

        self.denbins = self.params['denbins'].split(',')
        self.ndenbins = len(self.denbins)

        real_multipoles_fns = [self.params['real_multipoles_fn'].format(i)
                          for i in self.denbins]

        real_multipoles_fns = [
            self.params['real_multipoles_fn'].format(i)
            for i in self.denbins
        ]

        sv_rmu_filenames = [self.params['sv_rmu_filename'].format(i)
                            for i in self.denbins]

        if self.params['fit_data']:
            redshift_multipoles_fns = [
                self.params['redshift_multipoles_fn'].format(i)
                    for i in self.denbins
                ]

            smins = [int(i) for i in str(self.params['smin']).split(',')]
            smaxs = [int(i) for i in str(self.params['smax']).split(',')]

            redshift_multipoles_fn = {}
            real_multipoles_fn = {}
            sv_rmu_filename = {}
            self.smin = {}
            self.smax = {}

            for i, DS in enumerate(self.denbins):
                real_multipoles_fn[f'DS{DS}'] = real_multipoles_fns[i]
                sv_rmu_filename[f'DS{DS}'] = sv_rmu_filenames[i]
                if self.params['fit_data']:
                    redshift_multipoles_fn[f'DS{DS}'] = redshift_multipoles_fns[i]
                self.smin[f'DS{DS}'] = smins[i]
                self.smax[f'DS{DS}'] = smaxs[i]

            # cosmology for Minerva
            self.cosmo = Cosmology(omega_m=self.params['omega_m'])
            effective_z = self.params['effective_z']

            growth = self.cosmo.GrowthFactor(effective_z)
            f = self.cosmo.GrowthRate(effective_z)
            s8norm = self.params['sigma_8'] * growth
            self.bs8 = self.params['bias_mocks'] * s8norm
            self.fs8 = f * s8norm

            Hz = self.cosmo.HubbleParameter(effective_z)
            self.iaH = (1 + effective_z) / Hz

            if self.params['use_reconstruction']:
                self.beta_grid = np.genfromtxt(self.params['beta_grid_filename'])
                self.beta_prior = [self.beta_grid.min(), self.beta_grid.max()]
            else:
                self.beta_prior = [0.1, 0.8]

            if self.params['fit_data']:
                # read covariance matrix
                if os.path.isfile(self.params['covmat_fn']):
                    data = np.load(self.params['covmat_fn'], allow_pickle=True)
                    nbins = len(self.cov)
                    nmocks = data['nmocks']
                    self.cov = data['covmat']
                    if self.params['use_hartlap']:
                        hartlap = (1 - (nbins + 1) / (nmocks - 1))
                        self.icov = hartlap * np.linalg.inv(self.cov)
                    else:
                        self.icov = np.linalg.inv(self.cov)
                else:
                    raise FileNotFoundError('Covariance matrix not found.')

            if self.params['velocity_coupling'] not in ['empirical', 'linear']:
                raise ValueError("Only 'linear' or 'empirical' density-velocity"
                                 " couplings are supported.")

            self.r_for_xi = {}
            self.s_for_xi = {}
            self.mu_for_xi = {}
            self.xi_r_array = {}
            self.xi_0_array = {}
            self.xi_2_array = {}
            self.xi_4_array = {}
            self.scale_range = {}

            self.r_for_v = {}
            self.mu_for_v = {}
            self.sv_rmu_array = {}

            self.datavec = np.array([])

            for DS in self.denbins:
                denbin = f'DS{DS}'

                if self.params['use_reconstruction']:
                    data = np.load(real_multipoles_fn[denbin],
                                   allow_pickle=True)
                    self.r_for_xi[denbin] = data['r_c']
                    self.xi_r_array[denbin] = data['xi_0']

                    if self.params['constant_dispersion']:
                        self.r_for_v[denbin] = self.r_for_xi[denbin]
                        self.mu_for_v[denbin] = np.linspace(-1, 1, 80)
                        self.sv_rmu_array[denbin] = np.ones([len(self.r_for_v[denbin]),
                            len(self.mu_for_v[denbin])])
                    else:
                        data = np.load(sv_rmu_filename[denbin],
                                       allow_pickle=True)
                        self.r_for_v[denbin] = data[0]
                        self.mu_for_v[denbin] = data[1]
                        self.sv_rmu_array[denbin] = data[2]

                    if self.params['fit_data']:
                        data = np.load(redshift_multipoles_fn[denbin],
                                       allow_pickle=True)
                        self.s_for_xi[denbin] = data['r_c']
                        self.mu_for_xi[denbin] = data['mu_c']
                        self.xi_0_array[denbin] = data['xi_0']
                        self.xi_2_array[denbin] = data['xi_2']
                        self.xi_4_array[denbin] = data['xi_4']
                else:
                    data = np.load(real_multipoles_fn[denbin], allow_pickle=True)
                    self.r_for_xi[denbin] = data['r_c']
                    self.xi_r_array[denbin] = data['xi_0']

                    if self.params['constant_dispersion']:
                        self.r_for_v[denbin] = self.r_for_xi[denbin]
                        self.mu_for_v[denbin] = np.linspace(-1, 1, 80)
                        self.sv_rmu_array[denbin] = np.ones([len(self.r_for_v[denbin]),
                            len(self.mu_for_v[denbin])])
                    else:
                        self.r_for_v[denbin], self.mu_for_v[denbin], \
                            self.sv_rmu_array[denbin] = \
                            read_2darray(sv_rmu_filename[denbin])

                    if self.params['fit_data']:
                        data = np.load(redshift_multipoles_fn[denbin], allow_pickle=True)
                        self.s_for_xi[denbin] = data['r_c']
                        self.mu_for_xi[denbin] = data['mu_c']
                        self.xi_0_array[denbin] = data['xi_0']
                        self.xi_2_array[denbin] = data['xi_2']
                        self.xi_4_array[denbin] = data['xi_4']

                if self.params['fit_data']:
                    # restrict measured vectors to the desired fitting scales
                    self.scale_range[denbin] = (
                        self.s_for_xi[denbin] >=
                        self.smin[denbin]
                    ) & (
                        self.s_for_xi[denbin] <=
                        self.smax[denbin]
                    )

                xi_0, xi_2, xi_4 = self.data_multipoles(0.0, denbin)


            if self.params['velocity_coupling'] not in ['empirical', 'linear']:
                raise ValueError("Only 'linear' or 'empirical' "
                    "density-velocity couplings are supported.")


    def theory_multipoles(
        self, fs8, bs8,
        sigma_v, q_perp, q_para,
        s, mu, nu, denbin
    ):
        '''
        Calculates the redshift-space cross-correlation function
        from theory.
        '''
        beta = fs8 / bs8
        if self.params['velocity_coupling'] == 'linear':
            nu = 0.0

        if self.params['use_reconstruction']:
            # interpolate to get the DS model ingredients
            # corrected by AP distortions
            xi_r = self.interpolate_xi_r(beta, denbin)
            sv_rmu = self.interpolate_sv_rmu(beta, denbin)
        else:
            xi_r = self.xi_r_array[denbin]
            sv_rmu = self.sv_rmu_array[denbin]

        sv_rmu /= sv_rmu[-1, -1]
        r = self.r_for_xi[denbin]

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(r)
        dr = np.diff(r)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1. / (r[i] + dr / 2) ** 3 * \
                (np.sum(xi_r[:i+1] * ((r[:i+1] + dr / 2) ** 3
                 - (r[:i+1] - dr / 2) ** 3)))

        xi_r = InterpolatedUnivariateSpline(
            self.r_for_xi[denbin],
            xi_r, k=3, ext=0
        )
        int_xi_r = InterpolatedUnivariateSpline(
            self.r_for_xi[denbin],
            int_xi_r, k=3, ext=0
        )
        sv_rmu = RectBivariateSpline(
            self.r_for_v[denbin],
            self.mu_for_v[denbin],
            sv_rmu
        )

        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.params['rescale_rspace']:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r

        else:
            x = r

        if self.params['rescale_bs8']:
            y1 = (bs8 / self.bs8)**1 * xi_r(r)
            y2 = (bs8 / self.bs8)**1 * int_xi_r(r)
        else:
            y1 = xi_r(r)
            y2 = int_xi_r(r)

        y3 = sv_rmu(r, self.mu_for_v[denbin])

        # build new functions for the model ingredients after
        # correcting for the bias and Alcock-Paczynski distortions
        corrected_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        corrected_int_xi_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        corrected_sv_rmu = RectBivariateSpline(x, self.mu_for_v[denbin], y3)

        # rescale the line-of-sight velocity dispersion amplitude
        # at large scales
        sigma_v = q_para * sigma_v
        # reshape separation vectors to account for all possible
        # combinations
        y_grid = np.linspace(-5, 5, 500)
        S, MU, Y = np.meshgrid(s, mu, y_grid)

        # account for AP distortions
        corrected_sperp = S * np.sqrt(1 - MU ** 2) * q_perp
        corrected_spara = S * MU * q_para

        if self.params['rsd_model'] == 'streaming':
            # our integration variable "y" will span 5 sigma
            # around the mean of the Gaussian PDF. Since we
            # don't know the mean nor the dispersion apriori,
            # we will start with sigma_v = 500 around a zero-mean.
            sy = 500 * self.iaH
            y = Y * sy

            # all the separation vectors will now depend on the
            # integration variable through this relation
            rpara = corrected_spara - y

            # we now update the integration variable by calculating
            # the correct mean and dispersion in an iterative way.
            # 5 iterations is enough for convergence.
            for i in range(5):
                rr = np.sqrt(corrected_sperp ** 2 + rpara ** 2)
                mur = rpara / rr
                v_r = radial_velocity(
                    rr, corrected_xi_r, corrected_int_xi_r,
                    fs8, bs8, nu, 'cross-correlation'
                )
                y = Y * sy - v_r * mur
                rpara = corrected_spara - y

            sy = sigma_v * corrected_sv_rmu.ev(rr, mur) * self.iaH
            vrmu = v_r * mur

            # now calculate the Gaussian PDF with the correct mean
            # and dispersion
            los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

            # solve the streaming model integral
            integrand = los_pdf * (1 + corrected_xi_r(rr))
            xi_smu = (simps(integrand, y) - 1).T


        if self.params['rsd_model'] == 'kaiser':
            raise RuntimeError('Kaiser model not implemented.')

        xi_0, xi_2, xi_4 = multipole([0, 2, 4], s, mu, xi_smu)

        return xi_0, xi_2, xi_4

    def theory_xi_sigmapi(
        self, fs8, bs8,
        sigma_v, q_perp, q_para,
        s_perp, s_para, nu, denbin,
        squared=False, log=False
    ):
        '''
        Calculates the redshift-space cross-correlation function
        from theory.
        '''
        beta = fs8 / bs8

        if self.params['use_reconstruction']:
            # interpolate to get the DS model ingredients
            # corrected by AP distortions
            xi_r = self.interpolate_xi_r(beta, denbin)
            sv_rmu = self.interpolate_sv_rmu(beta, denbin)
        else:
            xi_r = self.xi_r_array[denbin]
            sv_rmu = self.sv_rmu_array[denbin]

        sv_rmu /= sv_rmu[-1, -1]
        r = self.r_for_xi[denbin]

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(r)
        dr = np.diff(r)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1. / (r[i] + dr / 2) ** 3 * \
                (np.sum(xi_r[:i+1] * ((r[:i+1] + dr / 2) ** 3
                 - (r[:i+1] - dr / 2) ** 3)))

        xi_r = InterpolatedUnivariateSpline(
            self.r_for_xi[denbin],
            xi_r, k=3, ext=0
        )
        int_xi_r = InterpolatedUnivariateSpline(
            self.r_for_xi[denbin],
            int_xi_r, k=3, ext=0
        )
        sv_rmu = RectBivariateSpline(
            self.r_for_v[denbin],
            self.mu_for_v[denbin],
            sv_rmu
        )

        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.params['rescale_rspace']:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r

        else:
            x = r

        if self.params['rescale_bs8']:
            y1 = (bs8 / self.bs8)**1 * xi_r(r)
            y2 = (bs8 / self.bs8)**1 * int_xi_r(r)
        else:
            y1 = xi_r(r)
            y2 = int_xi_r(r)

        y3 = sv_rmu(r, self.mu_for_v[denbin])

        # build new functions for the model ingredients after
        # correcting for the bias and Alcock-Paczynski distortions
        corrected_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        corrected_int_xi_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        corrected_sv_rmu = RectBivariateSpline(x, self.mu_for_v[denbin], y3)

        # rescale the line-of-sight velocity dispersion amplitude
        # at large scales
        sigma_v = q_para * sigma_v
        # reshape separation vectors to account for all possible
        # combinations
        y_grid = np.linspace(-5, 5, 500)
        S_PERP, S_PARA, Y = np.meshgrid(s_perp, s_para, y_grid)

        # account for AP distortions
        corrected_sperp = S_PERP * q_perp
        corrected_spara = S_PARA * q_para

        if self.params['rsd_model'] == 'streaming':
            # our integration variable "y" will span 5 sigma
            # around the mean of the Gaussian PDF. Since we
            # don't know the mean nor the dispersion apriori,
            # we will start with sigma_v = 500 around a zero-mean.
            sy = 500 * self.iaH
            y = Y * sy

            # all the separation vectors will now depend on the
            # integration variable through this relation
            rpara = corrected_spara - y

            # we now update the integration variable by calculating
            # the correct mean and dispersion in an iterative way.
            # 5 iterations is enough for convergence.
            for i in range(5):
                rr = np.sqrt(corrected_sperp ** 2 + rpara ** 2)
                mur = rpara / rr
                v_r = radial_velocity(
                    rr, corrected_xi_r, corrected_int_xi_r,
                    fs8, bs8, nu, 'cross-correlation'
                )
                y = Y * sy - v_r * mur
                rpara = corrected_spara - y

            sy = sigma_v * corrected_sv_rmu.ev(rr, mur) * self.iaH
            vrmu = v_r * mur

            # now calculate the Gaussian PDF with the correct mean
            # and dispersion
            los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

            # solve the streaming model integral
            integrand = los_pdf * (1 + corrected_xi_r(rr))
            xi_sigmapi = (simps(integrand, y) - 1)

            if squared:
                xi_sigmapi = xi_sigmapi * (s_perp ** 2 + s_para ** 2)
            if log:
                xi_sigmapi = np.log10(xi_sigmapi + 1)

        if self.params['rsd_model'] == 'kaiser':
            raise RuntimeError('Kaiser model not implemented.')

        return xi_sigmapi

    def data_multipoles(self, beta, denbin):
        if self.params['use_reconstruction']:
            xi_0, xi_2, xi_4 = self.interpolate_multipoles(beta, denbin)
        else:
            xi_0 = self.xi_0_array[denbin]
            xi_2 = self.xi_2_array[denbin]
            xi_4 = self.xi_4_array[denbin]

        xi_0 = xi_0[self.scale_range[denbin]]
        xi_2 = xi_2[self.scale_range[denbin]]
        xi_4 = xi_4[self.scale_range[denbin]]

        return xi_0, xi_2, xi_4

    def interpolate_xi_r(self, beta, denbin):
        xi_r = np.zeros(len(self.r_for_xi[denbin]))
        for i in range(len(self.r_for_xi[denbin])):
            interpolator = InterpolatedUnivariateSpline(
                self.beta_grid,
                self.xi_r_array[denbin][:, i], k=3, ext=0
            )
            xi_r[i] = interpolator(beta)
        return xi_r

    def interpolate_multipoles(self, beta, denbin):
        xi_0 = np.zeros(len(self.s_for_xi[denbin]))
        xi_2 = np.zeros(len(self.s_for_xi[denbin]))
        xi_4 = np.zeros(len(self.s_for_xi[denbin]))
        for i in range(len(self.s_for_xi[denbin])):
            interpolator_xi0 = InterpolatedUnivariateSpline(
                self.beta_grid, self.xi_0_array[denbin][:, i], k=3, ext=0
            )
            interpolator_xi2 = InterpolatedUnivariateSpline(
                self.beta_grid, self.xi_2_array[denbin][:, i], k=3, ext=0
            )
            interpolator_xi4 = InterpolatedUnivariateSpline(
                self.beta_grid, self.xi_4_array[denbin][:, i], k=3, ext=0
            )
            xi_0[i] = interpolator_xi0(beta)
            xi_2[i] = interpolator_xi2(beta)
            xi_4[i] = interpolator_xi4(beta)
        return xi_0, xi_2, xi_4

    def interpolate_sv_rmu(self, beta, denbin):
        sv_rmu = np.zeros([len(self.r_for_v[denbin]),
                          len(self.mu_for_v[denbin])])
        for i in range(len(self.r_for_v[denbin])):
            for j in range(len(self.mu_for_v[denbin])):
                interpolator = InterpolatedUnivariateSpline(
                    self.beta_grid,
                    self.sv_rmu_array[denbin][:, i, j],
                    k=3, ext=0
                )
                sv_rmu[i, j] = interpolator(beta)
        return sv_rmu

    def interpolate_xi_smu(self, beta, denbin):
        xi_smu = np.zeros([len(self.s_for_xi[denbin]),
                          len(self.mu_for_xi[denbin])])
        for i in range(len(self.s_for_xi[denbin])):
            for j in range(len(self.mu_for_xi[denbin])):
                interpolator = InterpolatedUnivariateSpline(
                    self.beta_grid,
                    self.xi_smu_array[denbin][:, i, j],
                    k=3, ext=0
                )
                xi_smu[i, j] = interpolator(beta)
        return xi_smu

    def log_likelihood(self, theta):
        """
        Log-likelihood for the RSD & AP fit.
        """
        fs8, bs8, sigma_v, q_perp, q_para, *nulist = theta

        nu = {}
        modelvec = np.array([])
        datavec = np.array([])

        for j, DS in enumerate(self.denbins):
            denbin = f'DS{DS}'
            nu[denbin] = nulist[j]

            s = self.s_for_xi[denbin][self.scale_range[denbin]]
            mu = self.mu_for_xi[denbin]

            xi_0, xi_2, xi_4 = self.theory_xi_smu(
                fs8,
                bs8,
                sigma_v,
                q_perp,
                q_para,
                s,
                mu,
                nu[denbin],
                denbin
            )

            poles = self.params['multipoles']

            if poles == '0,2,4':
                modelvec = np.concatenate((modelvec,
                                           xi_0,
                                           xi_2,
                                           xi_4))
            elif poles == '0,2':
                modelvec = np.concatenate((modelvec,
                                           xi_0,
                                           xi_2))
            elif poles == '2':
                modelvec = np.concatenate((modelvec,
                                           xi_2))
            else:
                raise ValueError("Unrecognized multipole fitting choice.")

            # build data vector
            beta = fs8 / bs8
            xi_0, xi_2, xi_4 = self.data_multipoles(beta, denbin)

            if poles == '2':
                datavec = np.concatenate(
                    (datavec, xi_2)
                )
            if poles == '0,2':
                datavec = np.concatenate(
                    (datavec, xi_0, xi_2)
                )
            if poles == '0,2,4':
                datavec = np.concatenate(
                    (datavec, xi_0, xi_2, xi_4)
                )

        chi2 = np.dot(np.dot((modelvec - datavec),
                             self.icov), modelvec - datavec)

        loglike = -0.5 * chi2

        return loglike

    def log_prior(self, theta):
        """
        Priors for the RSD & AP parameters.
        """

        fs8, bs8, sigma_v, q_perp, q_para, *nu = theta
        beta = fs8 / bs8
        nu = np.asarray(nu)

        if self.beta_prior[0] < beta < self.beta_prior[1] \
                and 0.1 < fs8 < 2.0 and 10 < sigma_v < 700 \
                and 0.1 < bs8 < 3.0 and 0.8 < q_perp < 1.2 \
                and 0.8 < q_para < 1.2 and np.all((-5 < nu) & (nu < 5)):
            return 0.0

        return -np.inf
