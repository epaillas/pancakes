import numpy as np
import sys
import os
from ..utilities.utilities import multipole, read_2darray
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.special import eval_legendre
from gsm.perturbation_theory.empirical import radial_velocity


class TwoPointCF:
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

        # read real-space galaxy monopole
        data = np.genfromtxt(self.params['xi_r_filename'])
        self.r_for_xi = data[:, 0]
        xi_r = data[:, 1]
        self.xi_r = InterpolatedUnivariateSpline(
            self.r_for_xi, xi_r, k=3
        )

        # read line-of-sight velocity dispersion profile 
        self.r_for_v, self.mu_for_v, sv_los = \
            read_2darray(self.params['sv_los_filename']
        self.sv_los = RectBivariateSpline(
            self.r_for_v, self.mu_for_v, sv_los,
            kx=1, ky=1
        )
        
        if self.params['fit_data']:
            # read redshift-space correlation function
            self.s_for_xi, self.mu_for_xi, xi_smu = \
                read_2darray(self.params['xi_smu_filename']

        # read covariance matrix
        if os.path.isfile(self.params['covmat_filename']):
            data = np.load(self.covmat_filename, allow_pickle=True)
            self.cov = data[0]
            nmocks = data[1]
            nbins = len(self.cov)
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

        # restrict to desired fitting scales
        self.scale_range = (self.s_for_xi >= self.params['smin']) & (
            self.s_for_xi <= self.params['smax'])

        # read data multipoles
        self.xi_0, self.xi_2, self.xi_4 = multipole(
            [0, 2, 4], self.s_for_xi, self.mu_for_xi,
            self.xi_smu
        )


    def theory_xi_smu(
        self, fs8, bs8,
        sigma_v, q_perp, q_para,
        s, mu, nu
    ):
        """
        Calculates the redshift-space cross-correlation function
        from theory.
        """
        beta = fs8 / bs8
        if self.params['velocity_coupling'] == 'linear':
            nu = 0.0
  
        # use the default model ingredients 
        xi_r = self.xi_r
        sv_los = self.sv_los
        r = self.r_for_xi

        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.params['rescale_rspace']:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r
        else:
            x = r

        # since we measure xi_r from a simulation with a fixed bs8,
        # we need to re-scale the amplitude of xi_r by the ratio
        # of the current bs8 and the one from simulation
        if self.params['rescale_bs8']: 
            y1 = (bs8 / self.bs8) * xi_r(r)
        else:
            y1 = xi_r(r)
        y2 = sv_los(r, self.mu_for_v)

        # build new functions for the model ingredients after
        # correcting for the bias and Alcock-Paczynski distortions
        corrected_xi_r = InterpolatedUnivariateSpline(x, y1, k=3)
        corrected_sv_los = RectBivariateSpline(x, self.mu_for_v, y2, kx=1, ky=1)

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(x)
        dx = np.diff(x)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1. / (x[i] + dx / 2) ** 3 * \
                (np.sum(y1(x)[:i+1] * ((x[:i+1] + dx / 2) ** 3 \
                - (x[:i+1] - dx / 2) ** 3)))
        corrected_int_xi_r = InterpolatedUnivariateSpline(x, int_xi_r, k=3)

        # rescale the line-of-sight velocity dispersion amplitude
        sigma_v *= q_para 

        # reshape separation vectors to account for all possible combinations
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
                    fs8, bs8, nu, 'autocorrelation'
                )
                y = Y * sy - v_r * mur
                rpara = corrected_spara - y

            sy = sigma_v * corrected_sv_los.ev(rr, mur) * self.iaH
            vrmu = v_r * mur

            # now calculate the Gaussian PDF with the correct mean
            # and dispersion
            los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

            # solve the streaming model integral
            integrand = los_pdf * (1 + corrected_xi_r(rr))
            xi_smu = (simps(integrand, y) - 1).T

        if self.params['rsd_model'] == 'kaiser':
            raise RuntimeError('Kaiser model not implemented.')

        return xi_smu 


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

        # use the default model ingredients 
        xi_r = self.xi_r
        sv_los = self.sv_los
        r = self.r_for_xi

        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.params['rescale_rspace']:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r
        else:
            x = r

        # since we measure xi_r from a simulation with a fixed bs8,
        # we need to re-scale the amplitude of xi_r by the ratio
        # of the current bs8 and the one from simulation
        if self.params['rescale_bs8']: 
            y1 = (bs8 / self.bs8) * xi_r(r)
        else:
            y1 = xi_r(r)
        y2 = sv_los(r, self.mu_for_v)

        # build new functions for the model ingredients after
        # correcting for the bias and Alcock-Paczynski distortions
        corrected_xi_r = InterpolatedUnivariateSpline(x, y1, k=3)
        corrected_sv_los = RectBivariateSpline(x, self.mu_for_v, y2, kx=1, ky=1)

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(x)
        dx = np.diff(x)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1. / (x[i] + dx / 2) ** 3 * \
                (np.sum(y1(x)[:i+1] * ((x[:i+1] + dx / 2) ** 3 \
                - (x[:i+1] - dx / 2) ** 3)))
        corrected_int_xi_r = InterpolatedUnivariateSpline(x, int_xi_r, k=3)

        # rescale the line-of-sight velocity dispersion amplitude
        sigma_v *= q_para 

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
                    fs8, bs8, nu, 'autocorrelation'
                )
                y = Y * sy - v_r * mur
                rpara = corrected_spara - y

            sy = sigma_v * corrected_sv_los.ev(rr, mur) * self.iaH
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

    def log_likelihood(self, theta):
        """
        Log-likelihood for the RSD & AP fit.
        """
        fs8, bs8, sigma_v, q_perp, q_para, nu = theta
        
        # build model vector
        model_xi = self.theory_xi_smu(fs8,
            bs8,
            sigma_v,
            q_perp,
            q_para,
            self.s_for_xi[self.scale_range],
            self.mu_for_xi,
            nu
        )

        xi_0, xi_2, xi_4 = get_multipoles(
            [0, 2, 4], self.s_for_xi[self.scale_range],
            self.mu_for_xi, model_xi
        )

        poles = self.params['multipoles']

        if poles == '0+2':
            modelvec = np.concatenate((xi_0, xi_2))
        if poles == '0+2+4':
            modelvec = np.concatenate((xi_0, xi_2, xi_4))
        if poles == '2':
            modelvec = xi_2

        # build data vector
        beta = fs8 / bs8
        xi_0, xi_2, xi_4 = self.multipoles_data(beta)

        if poles == '0+2':
            datavec = np.concatenate((xi_0, xi_2))
        elif poles == '0+2+4':
            datavec = np.concatenate((xi_0, xi_2, xi_4))
        elif poles == '2':
            datavec = xi_2
        else:
            raise ValueError("Unrecognized multipole fitting choice".)

        # calculate chi-sq and log-likelihood
        chi2 = np.dot(np.dot((modelvec - datavec),
            self.icov), modelvec - datavec)

        loglike = -0.5 * chi2

        return loglike

    def log_prior(self, theta):
        """
        Priors for the RSD & AP parameters.
        """
        fs8, bs8, sigma_v, q_perp, q_para, nu = theta
        beta = fs8 / bs8
            
        if self.beta_prior[0] < beta < self.beta_prior[1] \
                and 0.1 < fs8 < 2.0 and 10 < sigma_v < 700 \
                and 0.1 < bs8 < 3.0 and 0.8 < q_perp < 1.2 \
                and 0.8 < q_para < 1.2 and -5 < nu < 5:
            return 0.0

        return -np.inf
