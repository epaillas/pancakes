import numpy as np
import sys
import os
from utilities import Cosmology, Utilities
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.special import eval_legendre
from gsm.perturbation_theory.empirical import radial_velocity

class DensitySplitCCF:
    """
    Class to perform RSD & AP fits to the Density Split cross-correlation
    functons as in https://arxiv.org/abs/2101.09854
    """

    def __init__(
        self,
        xi_r_filename,
        xi_smu_filename,
        covmat_filename=None,
        sv_rmu_filename=None,
        betagrid_filename=None,
        poles='0+2',
        smin=0,
        smax=150,
        omega_m=0.285,
        sigma_8=0.828,
        effective_z=0.57,
        correlation_type='autocorr',
        velocity_coupling='empirical',
        independent_qs=0,
        rescale_rspace=0,
        use_hartlap=0,
        bias_mocks=2.07
    ):

        # filenames and options
        self.xi_r_filename = xi_r_filename
        self.sv_rmu_filename = sv_rmu_filename
        self.betagrid_filename = betagrid_filename
        self.xi_smu_filename = xi_smu_filename
        self.covmat_filename = covmat_filename
        self.smin = smin
        self.smax = smax
        self.velocity_coupling = velocity_coupling
        self.correlation_type = correlation_type
        self.poles = poles
        self.independent_qs = independent_qs
        self.rescale_rspace = rescale_rspace
        self.use_hartlap = use_hartlap

        # cosmology for Minerva
        self.omega_m = omega_m
        self.sigma_8 = sigma_8
        self.cosmo = Cosmology(h=1, omega_m=omega_m)
        self.effective_z = effective_z
        self.growth = self.cosmo.GrowthFactor(self.effective_z)
        self.f = self.cosmo.GrowthRate(self.effective_z)
        self.b = bias_mocks
        self.beta = self.f / self.b
        self.s8norm = self.sigma_8 * self.growth
        self.bs8 = self.b * self.s8norm
        self.fs8 = self.f * self.s8norm
        self.Hz = self.cosmo.HubbleParameter(self.effective_z)
        self.iaH = (1 + self.effective_z) / self.Hz

        if self.correlation_type == 'autocorr':
            print("Setting up redshift-space distortions model for autocorrelation function.")
        elif self.correlation_type == 'crosscorr':
            print('Setting up redshift-space distortions  model for cross-correlation function.')
        else:
            sys.exit('Invalid correlation type.')

        if self.poles == '0+2':
            print('MCMC will fit monopole and quadrupole.')
        elif self.poles == '0+2+4':
            print('MCMC will monopole, quadrupole and hexadecapole.')
        elif self.poles == '2':
            print('MCMC will only quadrupole.')
        else:
            sys.exit('Multipole fit not recognized.')

        if self.velocity_coupling == 'measured':
            sys.exit('Using measured radial velocity profile is not implemented.')
        elif self.velocity_coupling == 'empirical':
            print('Using empirical density-velocity coupling.')
        elif self.velocity_coupling == 'linear':
            print('Using linear density-velocity coupling')
        else:
            sys.exit('Invalid density-velocity coupling.')

        print('Fiducial cosmological parameters:')
        print('fs8 = {:1.5f}'.format(self.fs8))
        print('bs8 = {:1.5f}'.format(self.bs8))
        print('beta = {:1.5f}'.format(self.beta))
        print('f_zeff = {:1.5f}'.format(self.f))
        print('sigma_8_zeff = {:1.5f}'.format(self.s8norm))
        print('1/aH(zeff) = {:1.5f}'.format(self.iaH))

        # read real-space galaxy monopole
        data = np.load(self.xi_r_filename, allow_pickle=True)
        self.r_for_xi = data[0]
        self.xi_r_array = data[1]

        # read los velocity dispersion profile
        data = np.load(self.sv_rmu_filename, allow_pickle=True)
        self.r_for_v = data[0]
        self.mu_for_v = data[1]
        self.sv_rmu_array = data[2]

        # read beta grid
        self.betagrid = np.genfromtxt(self.betagrid_filename)
        self.beta_prior_range = [self.betagrid.min(), self.betagrid.max()]

        # read redshift-space correlation function
        data = np.load(self.xi_smu_filename, allow_pickle=True)
        self.s_for_xi = data[0]
        self.mu_for_xi = data[1]
        self.xi_smu_array = data[2]

        # read covariance matrix
        if os.path.isfile(self.covmat_filename):
            data = np.load(self.covmat_filename, allow_pickle=True)
            self.cov = data[0]
            self.nmocks = data[1]
            self.nbins = len(self.cov)
            if self.use_hartlap:
                hartlap = (1 - (self.nbins + 1) / (self.nmocks - 1))
                print(f'The Hartlap correction factor is {hartlap}')
                print(f'Number of mocks for covariance matrix: {self.nmocks}')
                self.icov = hartlap * np.linalg.inv(self.cov)
            else:
                self.icov = np.linalg.inv(self.cov)
        else:
            print(self.covmat_filename)
            sys.exit('Covariance matrix not found.')


        self.scale_range = (self.s_for_xi >= self.smin) & (
            self.s_for_xi <= self.smax)

    def multipoles_data(self, beta):
        xi_smu = self.interpolate_xi_smu(beta)
        s, xi_0 = Utilities.getMultipole(
            0, self.s_for_xi, self.mu_for_xi, xi_smu)
        s, xi_2 = Utilities.getMultipole(
            2, self.s_for_xi, self.mu_for_xi, xi_smu)
        s, xi_4 = Utilities.getMultipole(
            4, self.s_for_xi, self.mu_for_xi, xi_smu)

        xi_0 = xi_0[self.scale_range]
        xi_2 = xi_2[self.scale_range]
        xi_4 = xi_4[self.scale_range]

        return xi_0, xi_2, xi_4


    def theory_xi_smu(self, fs8, bs8, sigma_v, q_perp, q_para, s, mu, nu):
        '''
        This function calculates the theory multipoles using
        the Gaussian streaming model (Fisher 1995).
        '''
        beta = fs8 / bs8
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        corrected_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

  
        # interpolate to get the model ingredients
        # for the corresponding beta value
        xi_r = self.interpolate_xi_r(beta)
        sv_rmu = self.interpolate_sv_rmu(beta)
        sv_rmu /= sv_rmu[-1, -1]
      
        r = self.r_for_xi

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(r)
        dr = np.diff(r)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1./(r[i]+dr/2)**3 * (np.sum(xi_r[:i+1]*((r[:i+1]+dr/2)**3
              - (r[:i+1] - dr/2)**3)))

        xi_r = InterpolatedUnivariateSpline(self.r_for_xi, xi_r, k=3, ext=0)
        int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int_xi_r, k=3, ext=0)
        sv_rmu = RectBivariateSpline(self.r_for_v, self.mu_for_v, sv_rmu)

        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.rescale_rspace:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r
        else:
            x = r

        if self.bias_rescaling: 
            y1 = (bs8 / self.bs8) * xi_r(r)
            y2 = (bs8 / self.bs8) * int_xi_r(r)
        else:
            y1 = xi_r(r)
            y2 = int_xi_r(r)
        y3 = sv_rmu(r, self.mu_for_v)

        # build new functions for the model ingredients after
        # correcting for the bias and Alcock-Paczynski distortions
        corrected_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        corrected_int_xi_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        corrected_sv_rmu = RectBivariateSpline(x, self.mu_for_v, y3)

        # rescale the line-of-sight velocity dispersion amplitude at large scales
        sigma_v = q_para * sigma_v

        # reshape separation vectors to account for all possible combinations
        S = s.reshape(-1, 1)
        MU = mu.reshape(1, -1)

        # account for AP distortions
        corrected_sperp = S * np.sqrt(1 - MU ** 2) * q_perp
        corrected_spara = S * MU * q_para
        corrected_s = np.sqrt(corrected_spara ** 2 + corrected_sperp ** 2)
        corrected_mu = corrected_spara / corrected_s

        if self.model == 'streaming':
            # we build a variable that spans roughly -5 sigma to 5 sigma
            # of the line-of-sight velocity PDF at large scales,
            # but we express it as a distance
            sy = 500 * self.iaH
            y = np.linspace(-5 * sy, 5 * sy, 1000)
            
            # we substract this variable to the original line-of-sight
            # separation, such that it can also vary. We then compute
            # related quantities using this new separation
            rpara = corrected_spara - y
            rr = np.sqrt(corrected_sperp ** 2 + rpara ** 2)
            mur = rpara / rr
            sy = sigma_v * corrected_sv_rmu.ev(rr, mur) * self.iaH

            v_r = radial_velocity(rr, corrected_xi_r, corrected_int_xi_r,
                fs8, bs8, nu, self.correlation_type)
            
            # we calculate the PDF of the streaming model, using our original
            # integration variable, as well as the correct streaming velocity
            # and dispersion, which also depend on the integration variable
            los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

            # we construct the streaming model integrand and solve it numerically
            integrand = los_pdf * (1 + corrected_xi_r(rr))
            xi_smu = simps(integrand, y) - 1

        if self.model == 'kaiser':
            raise RunTimeError('Kaiser model not implemented.')

        return xi_smu 



    def interpolate_xi_r(self, beta):
        xi_r = np.zeros(len(self.r_for_xi))
        for i in range(len(self.r_for_xi)):
            interpolator = InterpolatedUnivariateSpline(self.betagrid, self.xi_r_array[:, i], k=3, ext=0)
            xi_r[i] = interpolator(beta)
        return xi_r

    def interpolate_sv_rmu(self, beta):
        sv_rmu = np.zeros([len(self.r_for_v), len(self.mu_for_v)])
        for i in range(len(self.r_for_v)):
            for j in range(len(self.mu_for_v)):
                interpolator = InterpolatedUnivariateSpline(self.betagrid, self.sv_rmu_array[:, i, j], k=3, ext=0)
                sv_rmu[i, j] = interpolator(beta)
        return sv_rmu

    def interpolate_xi_smu(self, beta):
        xi_smu = np.zeros([len(self.s_for_xi), len(self.mu_for_xi)])
        for i in range(len(self.s_for_xi)):
            for j in range(len(self.mu_for_xi)):
                interpolator = InterpolatedUnivariateSpline(self.betagrid, self.xi_smu_array[:, i, j], k=3, ext=0)
                xi_smu[i, j] = interpolator(beta)
        return xi_smu

    def log_likelihood(self, theta):

        if self.independent_qs:
            fs8, bs8, sigma_v, q_perp, q_para, nu = theta
        else:
            fs8, bs8, sigma_v, epsilon, nu = theta
            q = 1.0
            q_para = q * epsilon ** (-2/3)
            q_perp = epsilon * q_para
        
        # build model vector
        xi_0, xi_2, xi_4 = self.multipoles_theory(fs8,
                              bs8,
                              sigma_v,
                              q_perp,
                              q_para,
                              self.s_for_xi[self.scale_range],
                              self.mu_for_xi,
                              nu)

        if self.poles == '0+2':
            modelvec = np.concatenate((xi_0, xi_2))
        if self.poles == '0+2+4':
            modelvec = np.concatenate((xi_0, xi_2, xi_4))
        if self.poles == '2':
            modelvec = xi_2

        # build data vector
        beta = fs8 / bs8
        xi_0, xi_2, xi_4 = self.multipoles_data(beta)

        if self.poles == '0+2':
            datavec = np.concatenate((xi_0, xi_2))
        if self.poles == '0+2+4':
            datavec = np.concatenate((xi_0, xi_2, xi_4))
        if self.poles == '2':
            datavec = xi_2

        # calculate chi-sq and log-likelihood
        chi2 = np.dot(np.dot((modelvec - datavec),
          self.icov), modelvec - datavec)

        loglike = -0.5 * chi2

        return loglike

    def log_prior(self, theta):
        if self.independent_qs:
            fs8, bs8, sigma_v, q_perp, q_para, nu = theta
            beta = fs8 / bs8
            if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                and 0.1 < fs8 < 2.0 \
                and 10 < sigma_v < 700 \
                and 0.1 < bs8 < 3.0 \
                and 0.8 < q_perp < 1.2 \
                and 0.8 < q_para < 1.2 \
                and -10 < nu < 10:

                return 0.0
        else:
            fs8, bs8, sigma_v, epsilon, nu = theta
            beta = fs8 / bs8
            if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                and 0.1 < fs8 < 2.0 \
                and 0.1 < bs8 < 3.0 \
                and 10 < sigma_v < 700 \
                and 0.8 < epsilon < 1.2 \
                and -10 < nu < 10:
                return 0.0

        return -np.inf


class JointFit:
    '''
    Class for jointly fitting multiple CCFs
    calculated using the Density Split pipeline.
    The observable inputs are the redshift-space
    correlation functions in bins of s and mu.
    The class automatically decomposes the inputs
    in multipoles. The Gaussian streaming model
    is used to predict the theory multipoles for 
    each quantile.

    Input arguments:
        - xi_r_filename: type str, file containing the real space 
                         galaxy monopole or stacked density profile.
        - xi_smu_filename: type str, file containing the redshift-space
                           galaxy CCF, binned in s and mu.
        - covmat_filename: type str, file containing the covariance matrix
                           of the redshift-space correlation function.
        - sv_rmu_filename: type str, file containing the real space line-of-sight
                       velocity dispersion profile, binned in r and mu.
        - v_r_filename: type str, optional, file containing the real space 
                       radial velocity profile.
        - poles: type str, multipoles to fit, e.g. '0+2' for monopole and quadrupole,
                 '2' for quadrupole only, etc.
        - smin: type float, minimum redshift-space scale to fit.
        - sman: type float, maximum redshift-space scale to fit.
        - omega_m: type float, matter density parameter at a = 1.
        - sigma_8: type float, rms amplitude of linear power spectrum in spheres of 8 Mpc/h.
        - effective_z: type float, effective redshift of the galaxy sample.
        - correlation_type: type str, 'autocorr' or 'crosscorr'.
        - velocity_coupling: type str, 'empirical', 'linear' or 'measured'.
        - independent_qs: type int, 0 for fitting only AP ratio, 1 for fitting independent 
                              AP parameters (q_perp and q_para separately).
        - rescale_rspace: type int, 1 for making the model only sensitive to the AP ratio, 0 otherwise.
        - use_hartlap: type int, 1 for using Hartlap correction for covariance matrix, 0 otherwise.
    '''
    def __init__(self,
                 xi_r_filename,
                 xi_smu_filename,
                 betagrid_filename,
                 covmat_filename,
                 smin,
                 smax,
                 sv_rmu_filename=None,
                 poles='0+2',
                 omega_m=0.285,
                 sigma_8=0.828,
                 effective_z=0.57,
                 correlation_type='autocorr',
                 velocity_coupling='empirical',
                 independent_qs=0,
                 rescale_rspace=0,
                 rescale_bs8=1,
                 use_hartlap=0,
                 bias_mocks=2.07):

        self.correlation_type = correlation_type
        self.velocity_coupling = velocity_coupling
        self.use_hartlap = use_hartlap
        self.rescale_bs8 = rescale_bs8
        xi_r_filenames = xi_r_filename.split(',')
        sv_rmu_filenames = sv_rmu_filename.split(',')
        xi_smu_filenames = xi_smu_filename.split(',')
        smins = [int(i) for i in smin.split(',')]
        smaxs = [int(i) for i in smax.split(',')]

        self.ndenbins = len(xi_r_filenames)
        self.xi_r_filename = {}
        self.xi_smu_filename = {}
        self.sv_rmu_filename = {}
        self.smin = {}
        self.smax = {}
        self.covmat_filename = covmat_filename
        self.betagrid_filename = betagrid_filename

        for j in range(self.ndenbins):
            self.xi_r_filename['den{}'.format(j)] = xi_r_filenames[j]
            self.sv_rmu_filename['den{}'.format(j)] = sv_rmu_filenames[j]
            self.xi_smu_filename['den{}'.format(j)] = xi_smu_filenames[j]
            self.smin['den{}'.format(j)] = smins[j]
            self.smax['den{}'.format(j)] = smaxs[j]

        self.poles = poles
        self.independent_qs = independent_qs
        self.rescale_rspace = rescale_rspace

        if self.correlation_type == 'autocorr':
            print("Setting up RSD model for autocorrelation function.")
        elif self.correlation_type == 'crosscorr':
            print('Setting up RSD model for cross-correlation function.')
        else:
            sys.exit('Invalid correlation type.')
        print('Fitting {} density quantiles.'.format(self.ndenbins))

        # cosmology for Minerva
        self.omega_m = omega_m
        self.sigma_8 = sigma_8
        self.cosmo = Cosmology(omega_m=self.omega_m)
        self.effective_z = effective_z

        self.growth = self.cosmo.GrowthFactor(self.effective_z)
        self.f = self.cosmo.GrowthRate(self.effective_z)
        self.b = bias_mocks
        self.beta = self.f / self.b
        self.s8norm = self.sigma_8 * self.growth
        self.bs8 = self.b * self.s8norm
        self.fs8 = self.f * self.s8norm
        print('fs8 = {}'.format(self.fs8))
        print('bs8 = {}'.format(self.bs8))
        print('s8_zeff = {}'.format(self.s8norm))

        self.Hz = self.cosmo.HubbleParameter(self.effective_z)
        self.iaH = (1 + self.effective_z) / self.Hz

        self.betagrid = np.genfromtxt(self.betagrid_filename)
        self.beta_prior_range = [self.betagrid.min(), self.betagrid.max()]

        # read covariance matrix
        if os.path.isfile(self.covmat_filename):
            data = np.load(self.covmat_filename)
            self.cov = data[0]
            self.nmocks = data[1]
            self.nbins = len(self.cov)
            if self.use_hartlap:
                hartlap = (1 - (self.nbins + 1) / (self.nmocks - 1))
                print(f'The Hartlap correction factor is {hartlap}')
                print(f'Number of mocks for covariance matrix: {self.nmocks}')
                self.icov = hartlap * np.linalg.inv(self.cov)
            else:
                self.icov = np.linalg.inv(self.cov)
        else:
            print(self.covmat_filename)
            sys.exit('Covariance matrix not found.')

        if self.velocity_coupling == 'measured':
            sys.exit('Using measured radial velocity profile is not implemented.')
        elif self.velocity_coupling == 'empirical':
            print('Using empirical density-velocity coupling.')
        elif self.velocity_coupling == 'linear':
            print('Using linear density-velocity coupling')
        else:
            sys.exit('Invalid density-velocity coupling.')

        self.r_for_xi = {}
        self.s_for_xi = {}
        self.mu_for_xi = {}
        self.xi_r_array = {}
        self.xi_smu_array = {}
        self.scale_range = {}

        self.r_for_v = {}
        self.mu_for_v = {}
        self.sv_rmu_array = {}

        self.datavec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            # read real-space monopole
            data = np.load(self.xi_r_filename[denbin], allow_pickle=True)
            self.r_for_xi[denbin] = data[0]
            self.xi_r_array[denbin] = data[1]

            # read velocity dispersion
            data = np.load(self.sv_rmu_filename[denbin], allow_pickle=True)
            self.r_for_v[denbin] = data[0]
            self.mu_for_v[denbin] = data[1]
            self.sv_rmu_array[denbin] = data[2]

            # read redshift-space correlation function
            data = np.load(self.xi_smu_filename[denbin], allow_pickle=True)
            self.s_for_xi[denbin] = data[0]
            self.mu_for_xi[denbin] = data[1]
            self.xi_smu_array[denbin] = data[2]


            # restrict measured vectors to the desired fitting scales
            self.scale_range[denbin] = (self.s_for_xi[denbin] >= self.smin[denbin]) & (
                self.s_for_xi[denbin] <= self.smax[denbin])

    def multipoles_data(self, beta, denbin):
        xi_smu = self.interpolate_xi_smu(beta, denbin)
        s, xi_0 = Utilities.getMultipole(
            0, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu)
        s, xi_2 = Utilities.getMultipole(
            2, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu)
        s, xi_4 = Utilities.getMultipole(
            4, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu)

        xi_0 = xi_0[self.scale_range[denbin]]
        xi_2 = xi_2[self.scale_range[denbin]]
        xi_4 = xi_4[self.scale_range[denbin]]

        return xi_0, xi_2, xi_4

    def multipoles_theory(self, fs8, bs8, sigma_v, q_perp, q_para, s, mu, denbin, nu):
        '''
        This function calculates the theory multipoles using
        the Gaussian streaming model (Fisher 1995).
        '''
        beta = fs8 / bs8
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        corrected_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # interpolate to get the DS model ingredients
        # corrected by AP distortions
        xi_r = self.interpolate_xi_r(beta, denbin)
        sv_rmu = self.interpolate_sv_rmu(beta, denbin)
        sv_rmu /= sv_rmu[-1, -1]

        r = self.r_for_xi[denbin]

        # calculated integrated galaxy monopole
        int_xi_r = np.zeros_like(r)
        dr = np.diff(r)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1./(r[i]+dr/2)**3 * (np.sum(xi_r[:i+1]*((r[:i+1]+dr/2)**3
                                                                              - (r[:i+1] - dr/2)**3)))
        

        xi_r = InterpolatedUnivariateSpline(self.r_for_xi[denbin], xi_r, k=3, ext=0)
        int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi[denbin], int_xi_r, k=3, ext=0)
        sv_rmu = RectBivariateSpline(self.r_for_v[denbin], self.mu_for_v[denbin], sv_rmu)


        # this rescaling makes the model only sensitive
        # to the ratio of Alcock-Paczynski parameters
        if self.rescale_rspace:
            corrected_r = r * q_perp**(2/3) * q_para**(1/3)
            x = corrected_r

        else:
            x = r

        if self.rescale_bs8:
            # rescale the real-space monopoles according
            # to the linear bias from the mocks
            if self.correlation_type == 'autocorr':
                y1 = (bs8 / self.bs8)**2 * xi_r(r)
                y2 = (bs8 / self.bs8)**2 * int_xi_r(r)
            else:
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

        # rescale the line-of-sight velocity dispersion amplitude at large scales
        sigma_v = q_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                # we rescale the redshift-space separation vectors
                # according to the Alcock-Paczynski parameters
                corrected_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * q_perp
                corrected_spara = s[i] * mu[j] * q_para
                corrected_s = np.sqrt(corrected_spara ** 2. + corrected_sperp ** 2.)
                corrected_mu[j] = corrected_spara / corrected_s

                # we build a variable that spans roughly -5 sigma to 5 sigma
                # of the line-of-sight velocity PDF at large scales,
                # but we express it as a distance
                sy = 500 * self.iaH
                y = np.linspace(-5 * sy, 5 * sy, 1000)

                # we substract this variable to the original line-of-sight
                # separation, such that it can also vary. We then compute
                # related quantities using this new separation
                rpara = corrected_spara - y
                rr = np.sqrt(corrected_sperp ** 2 + rpara ** 2)
                mur = rpara / rr
                sy = sigma_v * corrected_sv_rmu.ev(rr, mur) * self.iaH

                # we figure out the density-velocity coupling depending
                # on the user's choice. Note that the expression is different
                # between 2PCF and CCF.
                if self.velocity_coupling == 'empirical':
                    if self.correlation_type == 'autocorr':
                        int2_xi_r = corrected_int_xi_r(rr) / (1 + corrected_xi_r(rr))
                        vrmu = -2/3 * beta * rr * int2_xi_r * \
                                (1 + nu * int2_xi_r) * mur
                    else:
                        vrmu = -1/3 * beta * rr * corrected_int_xi_r(rr) * mur \
                        / (1 + nu * corrected_xi_r(rr))
                else:
                    if self.correlation_type == 'autocorr':
                        vrmu = -2/3 * beta * rr * corrected_int_xi_r(rr) * mur \
                        / (1 + corrected_xi_r(rr)) 
                    else:
                        vrmu = -1/3 * beta * rr * corrected_int_xi_r(rr) * mur 

                # we calculate the PDF of the streaming model, using our original
                # integration variable, as well as the correct streaming velocity
                # and dispersion, which also depend on the integration variable
                los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

                # we construct the streaming model integrand and solve it numerically
                integrand = los_pdf * (1 + corrected_xi_r(rr))
                xi_model[j] = simps(integrand, y) - 1

            # build interpolating function for the model at corrected_s
            xi_model_func = InterpolatedUnivariateSpline(corrected_mu, xi_model, k=3, ext=0)

            # figure out the factor for the multipole conversion
            # depending on the mu range of the correlation function
            if corrected_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1

            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            ell = 0
            lmu = eval_legendre(ell, xaxis)
            yaxis = xi_model_func(xaxis) * (2 * ell + 1) / factor * lmu
            monopole[i] = simps(yaxis, xaxis)

            ell = 2
            lmu = eval_legendre(ell, xaxis)
            yaxis = xi_model_func(xaxis) * (2 * ell + 1) / factor * lmu
            quadrupole[i] = simps(yaxis, xaxis)

            ell = 4
            lmu = eval_legendre(ell, xaxis)
            yaxis = xi_model_func(xaxis) * (2 * ell + 1) / factor * lmu
            hexadecapole[i] = simps(yaxis, xaxis)

        return monopole, quadrupole, hexadecapole

    def interpolate_xi_r(self, beta, denbin):
        xi_r = np.zeros(len(self.r_for_xi[denbin]))
        for i in range(len(self.r_for_xi[denbin])):
            interpolator = InterpolatedUnivariateSpline(self.betagrid, self.xi_r_array[denbin][:, i], k=3, ext=0)
            xi_r[i] = interpolator(beta)
        return xi_r

    def interpolate_sv_rmu(self, beta, denbin):
        sv_rmu = np.zeros([len(self.r_for_v[denbin]), len(self.mu_for_v[denbin])])
        for i in range(len(self.r_for_v[denbin])):
            for j in range(len(self.mu_for_v[denbin])):
                interpolator = InterpolatedUnivariateSpline(self.betagrid, self.sv_rmu_array[denbin][:, i, j], k=3, ext=0)
                sv_rmu[i, j] = interpolator(beta)
        return sv_rmu

    def interpolate_xi_smu(self, beta, denbin):
        xi_smu = np.zeros([len(self.s_for_xi[denbin]), len(self.mu_for_xi[denbin])])
        for i in range(len(self.s_for_xi[denbin])):
            for j in range(len(self.mu_for_xi[denbin])):
                interpolator = InterpolatedUnivariateSpline(self.betagrid, self.xi_smu_array[denbin][:, i, j], k=3, ext=0)
                xi_smu[i, j] = interpolator(beta)
        return xi_smu

    def log_likelihood(self, theta):
        if self.independent_qs:

            if self.ndenbins == 2:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2 = theta
                nulist = [nu1, nu2]

            if self.ndenbins == 3:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3 = theta
                nulist = [nu1, nu2, nu3]

            if self.ndenbins == 4:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3, nu4 = theta
                nulist = [nu1, nu2, nu3, nu4]

            if self.ndenbins == 5:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3, nu4, nu5 = theta
                nulist = [nu1, nu2, nu3, nu4, nu5]

        else:

            if self.ndenbins == 2:
                fs8, bs8, sigma_v, epsilon, nu1, nu2 = theta
                nulist = [nu1, nu2]

            if self.ndenbins == 3:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3 = theta
                nulist = [nu1, nu2, nu3]

            if self.ndenbins == 4:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3, nu4 = theta
                nulist = [nu1, nu2, nu3, nu4]

            if self.ndenbins == 5:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3, nu4, nu5 = theta
                nulist = [nu1, nu2, nu3, nu4, nu5]

            q = 1.0
            q_para = q * epsilon ** (-2/3)
            q_perp = epsilon * q_para

        nu = {}
        modelvec = np.array([])
        datavec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            nu[denbin] = nulist[j]

            xi_0, xi_2, xi_4 = self.multipoles_theory(fs8,
                                                      bs8,
                                                      sigma_v,
                                                      q_perp,
                                                      q_para,
                                                      self.s_for_xi[denbin][self.scale_range[denbin]],
                                                      self.mu_for_xi[denbin],
                                                      denbin,
                                                      nu[denbin])

            if self.poles == '0+2+4':
                modelvec = np.concatenate((modelvec,
                                           xi_0,
                                           xi_2,
                                           xi_4))
            elif self.poles == '0+2':
                modelvec = np.concatenate((modelvec,
                                           xi_0,
                                           xi_2))
            elif self.poles == '2':
                modelvec = np.concatenate((modelvec,
                                           xi_2))

            # build data vector
            beta = fs8 / bs8
            xi_0, xi_2, xi_4 = self.multipoles_data(beta, denbin)

            if self.poles == '2':
                datavec = np.concatenate((datavec,
                    xi_2))
            if self.poles == '0+2':
                datavec = np.concatenate((datavec,
                    xi_0,
                    xi_2))
            if self.poles == '0+2+4':
                datavec = np.concatenate((datavec,
                    xi_0,
                    xi_2,
                    xi_4))

        chi2 = np.dot(np.dot((modelvec - datavec),
                             self.icov), modelvec - datavec)

        loglike = -0.5 * chi2

        return loglike

    def log_prior(self, theta):

        if self.independent_qs:
            if self.ndenbins == 2:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < q_perp < 1.2 \
                    and 0.8 < q_para < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10:
                    return 0.0

            if self.ndenbins == 3:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < q_perp < 1.2 \
                    and 0.8 < q_para < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10:
                    return 0.0

            if self.ndenbins == 4:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3, nu4 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < q_perp < 1.2 \
                    and 0.8 < q_para < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10 \
                    and -10 < nu4 < 10:
                    return 0.0

            if self.ndenbins == 5:
                fs8, bs8, sigma_v, q_perp, q_para, nu1, nu2, nu3, nu4, nu5 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < q_perp < 1.2 \
                    and 0.8 < q_para < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10 \
                    and -10 < nu4 < 10 \
                    and -10 < nu5 < 10:
                    return 0.0

        else:
            if self.ndenbins == 2:
                fs8, bs8, sigma_v, epsilon, nu1, nu2 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < epsilon < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10:
                    return 0.0

            if self.ndenbins == 3:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < epsilon < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10:
                    return 0.0

            if self.ndenbins == 4:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3, nu4 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < epsilon < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10 \
                    and -10 < nu4 < 10:
                    return 0.0

            if self.ndenbins == 5:
                fs8, bs8, sigma_v, epsilon, nu1, nu2, nu3, nu4, nu5 = theta
                beta = fs8 / bs8
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] \
                    and 0.1 < fs8 < 2.0 \
                    and 0.1 < bs8 < 3.0 \
                    and 10 < sigma_v < 700 \
                    and 0.8 < epsilon < 1.2 \
                    and -10 < nu1 < 10 \
                    and -10 < nu2 < 10 \
                    and -10 < nu3 < 10 \
                    and -10 < nu4 < 10 \
                    and -10 < nu5 < 10:
                    return 0.0

        return -np.inf
